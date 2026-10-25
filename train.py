import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
# train
if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options   把所有的参数都设置好并输出到checkpoints/opt.txt中
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options  把数据集根据参数设置好
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options  根据参数初始化了网络结构

    model.setup(opt)               # regular setup: load and print networks; create schedulers   设置了学习率，打印了参数量和网络结构（如果设置了参数要打印网络的话）
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots  初始化了保存图片和损失的文件夹
    total_iters = 0                # the total number of training iterations训练迭代的总数

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    #1，100+100+1 outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch   记录整个epoch的时间
        iter_data_time = time.time()    # timer for data loading per iteration   记录每次迭代数据加载的时间
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch  当前epoch中的训练迭代次数，每个epoch重置为0

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration  #记录每次迭代计算时间
            # 每过100张图片输出当前epoch迭代所花时间  opt.print_freq=100
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()   #重置是否保存self.saved=false
            total_iters += opt.batch_size  # opt.batch_size=1  总迭代数
            epoch_iter += opt.batch_size    #当前epoch的迭代数
            model.set_input(data)  # unpack data from dataset and apply preprocessing 设置了图片和标签以及图片的路径
            model.optimize_parameters(epoch)  # calculate loss functions, get gradients, update network weights   进行前向传播然后反向传播再用SGD调优

            # 2023年4月27日 每过display_freq(400次)张照片，就侧输出和fused层图片输出
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                #  每过update_html_freq(400次)张图片,保存训练结果到html
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()   #这个函数好像没啥用
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)  #将侧输出图片和fused图片输出到checkpoint/Deepcrack/images/web中

            # 每过print_freq(100次)，就print training losses and save logging information to the disk
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

            # 每过save_latest_freq(5000次)，就cache our latest model
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        #每过save_epoch_freq(50轮)saving best_pth
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
