# import torch.backends.cudnn as cudnn
import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = 0

for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    save_result = True
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        if model.is_skip():
            continue
        model.update_D(data)
        model.update_G()
        model.balance()

        if save_result or total_steps % opt.display_freq == 0:
            save_result = save_result or total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(
                model.get_current_visuals()[0], epoch, ncols=4, save_result=save_result)
            #visualizer.display_current_feature_results(
            #    model.get_current_visuals()[1:], epoch, ncols=1, save_result=save_result)
            save_result = False

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(
                    epoch_iter) / dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    model.update_learning_rate()
