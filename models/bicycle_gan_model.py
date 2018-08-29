import numpy as np
import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from  .heightmap_normals_loss import HeightmapNormalsLoss
import util.kdsutil as kdsutil


class BiCycleGANModel(BaseModel):
    def name(self):
        return 'BiCycleGANModel'

    def initialize(self, opt):
        if opt.isTrain:
            assert opt.batchSize % 2 == 0  # load two images at one time.

        use_D = opt.isTrain and opt.lambda_GAN > 0.0
        use_D2 = opt.isTrain and opt.lambda_GAN2 > 0.0 and not opt.use_same_D
        use_E = opt.isTrain or not opt.no_encode
        use_L2 = opt.isTrain and opt.use_L2
        BaseModel.initialize(self, opt)
        self.use_normals = self.opt.use_normals
        self.init_data(opt, use_D=use_D, use_D2=use_D2, use_E=use_E, use_vae=True)

        if self.use_normals:
            self.gen_normals = HeightmapNormalsLoss(self.opt.gpu_ids)
        self.skip = False
        self.mse_loss = torch.nn.MSELoss()



    def is_skip(self):
        return self.skip

    def forward(self):
        # get real images
        self.skip = self.opt.isTrain and self.input_A.size(0) < self.opt.batchSize
        if self.skip:
            print('skip this point data_size = %d' % self.input_A.size(0))
            return

        # THIS IS CRUDE.  The batch size must be even so it can use the odd pairs as "encoded" and the even as "random"
        half_size = self.opt.batchSize // 2  # floordiv operation
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        # A1, B1 for encoded; A2, B2 for random
        self.real_A_encoded = self.real_A[0:half_size]
        self.real_A_random = self.real_A[half_size:]
        self.real_B_encoded = self.real_B[0:half_size]
        self.real_B_random = self.real_B[half_size:]
        # get encoded z
        self.mu, self.logvar = self.netE.forward(self.real_B_encoded)
        std = self.logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
        self.z_encoded = eps.mul(std).add_(self.mu)
        # get random z
        self.z_random = self.get_z_random(self.real_A_random.size(0), self.opt.nz, 'gauss')
        # generate fake_B_encoded
        self.fake_B_encoded = self.netG.forward(self.real_A_encoded, self.z_encoded)
        # generate fake_B_random
        self.fake_B_random = self.netG.forward(self.real_A_encoded, self.z_random)  # notice it uses real_A_encoded as input not real_A_random
        if self.opt.conditional_D:   # tedious conditoinal data
            self.fake_data_encoded = torch.cat([self.real_A_encoded, self.fake_B_encoded], 1)
            self.real_data_encoded = torch.cat([self.real_A_encoded, self.real_B_encoded], 1)
            self.fake_data_random = torch.cat([self.real_A_encoded, self.fake_B_random], 1)
            self.real_data_random = torch.cat([self.real_A_random, self.real_B_random], 1)
        else:
            self.fake_data_encoded = self.fake_B_encoded
            self.fake_data_random = self.fake_B_random
            self.real_data_encoded = self.real_B_encoded
            self.real_data_random = self.real_B_random

        if self.use_normals:
            self.fake_normal_encoded = self.gen_normals(self.fake_B_encoded)
            self.fake_normal_random = self.gen_normals(self.fake_B_random)
            self.real_normal_encoded = self.gen_normals(self.real_B_encoded)
            self.real_normal_random = self.gen_normals(self.real_B_random)

        # compute z_predict
        if self.opt.lambda_z > 0.0:
            self.mu2, logvar2 = self.netE.forward(self.fake_B_random)  # mu2 is a point estimate

    def encode(self, input_data):
        mu, logvar = self.netE.forward(Variable(input_data, volatile=True))
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
        return eps.mul(std).add_(mu)

    def backward_D(self, netD, real, fake, encoded):
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake = netD.forward(fake.detach())
        if encoded:
            # TODO: Get features from fake here
            self.fake_relu_1 = netD.relu_1_x  # 64x128x128
            self.fake_relu_2 = netD.relu_2_x  # 128x64x64
            self.fake_relu_3 = netD.relu_3_x  # 256x32x32
            self.fake_relu_4 = netD.relu_4_x  # 512x31x31
        # real
        pred_real = netD.forward(real)
        if encoded:
            self.real_relu_1 = netD.relu_1_x  # 64x128x128
            self.real_relu_2 = netD.relu_2_x  # 128x64x64
            self.real_relu_3 = netD.relu_3_x  # 256x32x32
            self.real_relu_4 = netD.relu_4_x  # 512x31x31

        loss_D_fake, losses_D_fake = self.criterionGAN(pred_fake, False)
        loss_D_real, losses_D_real = self.criterionGAN(pred_real, True)
        # Combined loss
        loss_D = loss_D_fake + loss_D_real

        if self.opt.use_features and encoded:
            self.loss_feat_1 = self.mse_loss(self.real_relu_1.detach(), self.fake_relu_1.detach())
            self.loss_feat_2 = self.mse_loss(self.real_relu_2.detach(), self.fake_relu_2.detach())
            self.loss_feat_3 = self.mse_loss(self.real_relu_3.detach(), self.fake_relu_3.detach())
            self.loss_feat_4 = self.mse_loss(self.real_relu_4.detach(), self.fake_relu_4.detach())
            self.loss_feat = self.loss_feat_1 + self.loss_feat_2 + self.loss_feat_3 + self.loss_feat_4
            loss_D += self.opt.lambda_features * self.loss_feat

        loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real]

    def backward_G_GAN(self, fake, netD=None, ll=0.0):
        if ll > 0.0:
            pred_fake = netD.forward(fake)
            loss_G_GAN, losses_G_GAN = self.criterionGAN(pred_fake, True)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    # TODO: Add feature space loss here!!!!!!
    def backward_EG(self):
        # 1, G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(
            self.fake_data_encoded, self.netD, self.opt.lambda_GAN)
        if self.opt.use_same_D:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD, self.opt.lambda_GAN2)
        else:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD2, self.opt.lambda_GAN2)
        # 2. KL loss
        if self.opt.lambda_kl > 0.0:
            kl_element = self.mu.pow(2).add_(self.logvar.exp()).mul_(-1).add_(1).add_(self.logvar)
            self.loss_kl = torch.sum(kl_element).mul_(-0.5) * self.opt.lambda_kl
        else:
            self.loss_kl = 0
        # 3, reconstruction |fake_B-real_B|
        if self.opt.lambda_L1 > 0.0:
            if self.use_normals:
                self.loss_G_L1 = self.criterionL1(self.fake_normal_encoded, self.real_normal_encoded) * self.opt.lambda_L1
            else:
                self.loss_G_L1 = self.criterionL1(self.fake_B_encoded, self.real_B_encoded) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = 0.0

        self.loss_G = self.loss_G_GAN + self.loss_G_GAN2 + self.loss_G_L1 + self.loss_kl
        self.loss_G.backward(retain_graph=True)

    def update_D(self, data):
        self.set_requires_grad(self.netD, True)
        self.set_input(data)
        self.forward()
        if self.is_skip():
            return
        # update D1
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_data_encoded, self.fake_data_encoded, True)
            if self.opt.use_same_D:
                self.loss_D2, self.losses_D2 = self.backward_D(self.netD, self.real_data_random, self.fake_data_random, False)
            self.optimizer_D.step()

        if self.opt.lambda_GAN2 > 0.0 and not self.opt.use_same_D:
            self.optimizer_D2.zero_grad()
            self.loss_D2, self.losses_D2 = self.backward_D(self.netD2, self.real_data_random, self.fake_data_random, False)
            self.optimizer_D2.step()

    def backward_G_alone(self):
        # 3, reconstruction |(E(G(A, z_random)))-z_random|
        if self.opt.lambda_z > 0.0:
            self.loss_z_L1 = torch.mean(torch.abs(self.mu2 - self.z_random)) * self.opt.lambda_z
            self.loss_z_L1.backward()
        else:
            self.loss_z_L1 = 0.0

    def update_G(self):
        # update G and E
        self.set_requires_grad(self.netD, False)
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_EG()
        self.optimizer_G.step()
        self.optimizer_E.step()
        # update G only
        if self.opt.lambda_z > 0.0:
            self.optimizer_G.zero_grad()
            self.optimizer_E.zero_grad()
            self.backward_G_alone()
            self.optimizer_G.step()

    def get_current_errors(self):
        z1 = self.z_encoded.data.cpu().numpy()
        if self.opt.lambda_z > 0.0:
            loss_G = self.loss_G + self.loss_z_L1
        else:
            loss_G = self.loss_G
        ret_dict = OrderedDict([('z_encoded_mag', np.mean(np.abs(z1))),
                                ('G_total', loss_G.data[0])])

        if self.opt.lambda_L1 > 0.0:
            G_L1 = self.loss_G_L1.data[0] if self.loss_G_L1 is not None else 0.0
            ret_dict['G_L1_encoded'] = G_L1

        if self.opt.lambda_z > 0.0:
            z_L1 = self.loss_z_L1.data[0] if self.loss_z_L1 is not None else 0.0
            ret_dict['z_L1'] = z_L1

        if self.opt.lambda_kl > 0.0:
            ret_dict['KL'] = self.loss_kl.data[0]

        if self.opt.lambda_GAN > 0.0:
            ret_dict['G_GAN'] = self.loss_G_GAN.data[0]
            ret_dict['D_GAN'] = self.loss_D.data[0]

        if self.opt.lambda_GAN2 > 0.0:
            ret_dict['G_GAN2'] = self.loss_G_GAN2.data[0]
            ret_dict['D_GAN2'] = self.loss_D2.data[0]

        if self.opt.use_features and self.opt.lambda_GAN > 0.0:
            ret_dict['D_ENC_FEAT'] = self.loss_feat.data[0]

        return ret_dict

    def get_current_visuals(self):
        real_A_encoded = util.tensor2im(self.real_A_encoded.data)
        real_A_random = util.tensor2im(self.real_A_random.data)
        real_B_encoded = util.tensor2im(self.real_B_encoded.data)
        real_B_random = util.tensor2im(self.real_B_random.data)
        ret_dict = OrderedDict([('real_A_encoded', real_A_encoded), ('real_B_encoded', real_B_encoded),
                                ('real_A_random', real_A_random), ('real_B_random', real_B_random)])

        if self.opt.isTrain:
            fake_random = util.tensor2im(self.fake_B_random.data)
            fake_encoded = util.tensor2im(self.fake_B_encoded.data)
            ret_dict['fake_random'] = fake_random
            ret_dict['fake_encoded'] = fake_encoded
            if self.opt.use_normals:
                ret_dict['fake_normal_encoded'] = self.gen_normals.convert_normals_to_image(self.fake_normal_encoded)
                ret_dict['fake_normal_random'] = self.gen_normals.convert_normals_to_image(self.fake_normal_random)
                ret_dict['real_normal_encoded'] = self.gen_normals.convert_normals_to_image(self.real_normal_encoded)
                ret_dict['real_normal_random'] = self.gen_normals.convert_normals_to_image(self.real_normal_random)

            if self.opt.use_features or True:  # TODO: make using features an option
                im_fr1 = util.tensor2im(kdsutil.features2gridim(self.fake_relu_1)) # 64x128x128
                im_rr1 = util.tensor2im(kdsutil.features2gridim(self.real_relu_1)) # 64x128x128
                im_cr1 = util.tensor2im(kdsutil.features2gridim(torch.abs(self.real_relu_1 * self.fake_relu_1), normalize=True)) # 64x128x128
                im_fr2 = util.tensor2im(kdsutil.features2gridim(self.fake_relu_2)) # 128x64x64
                im_rr2 = util.tensor2im(kdsutil.features2gridim(self.real_relu_2)) # 128x64x64
                im_cr2 = util.tensor2im(kdsutil.features2gridim(torch.abs(self.real_relu_2 * self.fake_relu_2), normalize=True)) # 128x64x64
                im_fr3 = util.tensor2im(kdsutil.features2gridim(self.fake_relu_3)) # 256x32x32
                im_rr3 = util.tensor2im(kdsutil.features2gridim(self.real_relu_3)) # 256x32x32
                im_cr3 = util.tensor2im(kdsutil.features2gridim(torch.abs(self.real_relu_3 * self.fake_relu_3), normalize=True)) # 256x32x32
                im_fr4 = util.tensor2im(kdsutil.features2gridim(self.fake_relu_4)) # 512x31x31
                im_rr4 = util.tensor2im(kdsutil.features2gridim(self.real_relu_4)) # 512x31x31
                im_cr4 = util.tensor2im(kdsutil.features2gridim(torch.abs(self.real_relu_4 * self.fake_relu_4), normalize=True)) # 512x31x31
                r1_dict = OrderedDict()
                r2_dict = OrderedDict()
                r3_dict = OrderedDict()
                r4_dict = OrderedDict()
                r1_dict['fake feat relu1'] = im_fr1
                r1_dict['real feat relu1'] = im_rr1
                r1_dict['mod feat relu1'] = im_cr1
                r2_dict['fake feat relu2'] = im_fr2
                r2_dict['real feat relu2'] = im_rr2
                r2_dict['mod feat relu2'] = im_cr2
                r3_dict['fake feat relu3'] = im_fr3
                r3_dict['real feat relu3'] = im_rr3
                r3_dict['mod feat relu3'] = im_cr3
                r4_dict['fake feat relu4'] = im_fr4
                r4_dict['real feat relu4'] = im_rr4
                r4_dict['mod feat relu4'] = im_cr4


        return ret_dict, r1_dict, r2_dict, r3_dict, r4_dict

    def normalize_features(self, feats, eps=1e-10):
        print("Sum size: {}".format(sum.size()))
        sum = torch.sum(feats**2, dim=1)
        print("Sum size: {}".format(sum.size()))

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        if self.opt.lambda_GAN > 0.0:
            self.save_network(self.netD, 'D', label, self.gpu_ids)
        if self.opt.lambda_GAN2 > 0.0 and not self.opt.use_same_D:
            self.save_network(self.netD, 'D2', label, self.gpu_ids)
        self.save_network(self.netE, 'E', label, self.gpu_ids)
