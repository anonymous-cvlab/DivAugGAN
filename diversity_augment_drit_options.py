import argparse


class TrainOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    
    # data loader related
    self.parser.add_argument('--dataroot', type=str, required=True, help='path of data')
    self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
    self.parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    self.parser.add_argument('--resize_size', type=int, default=256, help='resized image size for training')
    self.parser.add_argument('--crop_size', type=int, default=216, help='cropped image size for training')
    self.parser.add_argument('--input_dim_a', type=int, default=3, help='# of input channels for domain A')
    self.parser.add_argument('--input_dim_b', type=int, default=3, help='# of input channels for domain B')
    self.parser.add_argument('--nThreads', type=int, default=8, help='# of threads for data loader')
    self.parser.add_argument('--no_flip', action='store_true', help='specified if no flipping')
    
    
    # output related
    self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
    self.parser.add_argument('--display_dir', type=str, default='../logs', help='path for saving display results')
    self.parser.add_argument('--result_dir', type=str, default='../results', help='path for saving result images and models')
    self.parser.add_argument('--display_freq', type=int, default=1, help='freq (iteration) of display')
    self.parser.add_argument('--img_save_freq', type=int, default=5, help='freq (epoch) of saving images')
    self.parser.add_argument('--model_save_freq', type=int, default=10, help='freq (epoch) of saving models')
    self.parser.add_argument('--no_display_img', action='store_true', help='specified if no dispaly')
    
    
    # training related
    self.parser.add_argument('--concat', type=int, default=1, help='concatenate attribute features for translation, set 0 for using feature-wise transform')
    self.parser.add_argument('--dis_scale', type=int, default=3, help='scale of discriminator')
    self.parser.add_argument('--dis_norm', type=str, default='None', help='normalization layer in discriminator [None, Instance]')
    self.parser.add_argument('--dis_spectral_norm', action='store_true', help='use spectral normalization in discriminator')
    self.parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')
    self.parser.add_argument('--n_ep', type=int, default=1200, help='number of epochs') # 400 * d_iter
    self.parser.add_argument('--n_ep_decay', type=int, default=600, help='epoch start decay learning rate, set -1 if no decay') # 200 * d_iter
    self.parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--d_iter', type=int, default=3, help='# of iterations for updating content discriminator')
    self.parser.add_argument('--gpu', type=int, default=0, help='gpu')
    
    # diversity augmentation loss related
    self.parser.add_argument('--no_diversity_augment', action='store_true', help='disable diversity augment regularization')
    self.parser.add_argument('--lambda_alpha', type=float, default=1.0, help='# weights of distance to a random pickup point')
    self.parser.add_argument('--lambda_scale_factor', type=int, default=10.0, help='# weights of the scale factor')
    self.parser.add_argument('--lambda_dr', type=float, default=1.0, help='# weights of dr term in diversity augmentation regularizer loss')
    self.parser.add_argument('--lambda_rvc', type=float, default=1.0, help='# weights of rvc term in diversity augmentation regularizer loss')
    
  
  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    return self.opt
  
    
class TestOptions(): 
  def __init__(self):   
    self.parser = argparse.ArgumentParser()
    
    # data loader related
    self.parser.add_argument('--dataroot', type=str, required=True, help='path of data')
    self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
    self.parser.add_argument('--resize_size', type=int, default=256, help='resized image size for training')
    self.parser.add_argument('--crop_size', type=int, default=216, help='cropped image size for training')
    self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    self.parser.add_argument('--nThreads', type=int, default=4, help='for data loader')
    self.parser.add_argument('--input_dim_a', type=int, default=3, help='# of input channels for domain A')
    self.parser.add_argument('--input_dim_b', type=int, default=3, help='# of input channels for domain B')
    self.parser.add_argument('--a2b', type=int, default=1, help='translation direction, 1 for a2b, 0 for b2a')

    
    # ouptput related
    self.parser.add_argument('--num', type=int, default=5, help='number of outputs per image')
    self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
    self.parser.add_argument('--result_dir', type=str, default='../outputs', help='path for saving result images and models')


    # model related
    self.parser.add_argument('--concat', type=int, default=1, help='concatenate attribute features for translation, set 0 for using feature-wise transform')
    self.parser.add_argument('--resume', type=str, required=True, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--gpu', type=int, default=0, help='gpu')
    
    # diversity augmentation loss related
    self.parser.add_argument('--no_diversity_augment', action='store_true', help='disable diversity augment regularization')
    self.parser.add_argument('--lambda_alpha', type=float, default=1.0, help='# weights of distance to a random pickup point')
    self.parser.add_argument('--lambda_scale_factor', type=int, default=10.0, help='# weights of the scale factor')
    self.parser.add_argument('--lambda_dr', type=float, default=1.0, help='# weights of dr term in diversity augmentation regularizer loss')
    self.parser.add_argument('--lambda_rvc', type=float, default=1.0, help='# weights of rvc term in diversity augmentation regularizer loss')
    

    # metrics related
    self.parser.add_argument('--fid_batch_size', type=int, default=64, help='Batch size for FID calculation')
    self.parser.add_argument('--prdc_batch_size', type=int, default=50, help='Batch size for Precision & Recall, and Density & Coverage computation')
    self.parser.add_argument('--k', type=int, default=3, help='Batch size to use')
    self.parser.add_argument('--nearest_k', type=int, default=5, help='nearest_k value')
    self.parser.add_argument('--prdc_num_samples', type=int, default=15000, help='number of samples to use Precision & Recall, and Density & Coverage computation')

  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    # set irrelevant options
    self.opt.dis_scale = 3
    self.opt.dis_norm = 'None'
    self.opt.dis_spectral_norm = False
    return self.opt        
  
  