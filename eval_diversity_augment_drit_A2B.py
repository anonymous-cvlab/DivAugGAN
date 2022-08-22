import os, sys
import numpy as np
import torch
from saver import save_images, save_imgs
import pdb
from diversity_augment_drit_options  import TestOptions
from dataset import dataset_single
from diversity_augment_drit_model import DiversityAugmentDRIT
from metrics.lpips import calculate_lpips_given_images
from metrics.fid_score import calculate_fid_given_paths
from metrics.prdc import PRDC



def calculate_fid_two_domains_unpaired(opts):
  if opts.a2b: 
    source_domain = 'A'
    target_domain = 'B'
    direction = 'A2B'
  else: 
    print("Need to be set opts.a2b to be 1")
    input()  
    
  pdb.set_trace()  
  real_source_train_path = os.path.join(opts.dataroot, 'train' + source_domain)  
  real_target_train_path = os.path.join(opts.dataroot, 'train' + target_domain)
  real_source_test_path = os.path.join(opts.dataroot, 'test' + source_domain)
  real_target_test_path = os.path.join(opts.dataroot, 'test' + target_domain)
  
  fake_source2target_path = os.path.join(opts.result_dir, opts.name, 'fake_{direction:s}'.format(direction=direction))
  task = "{task_name:s}-{source_domain:s}2{target_domain:s}".format(task_name=opts.name, source_domain=source_domain, target_domain=target_domain)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('Calculating FID for {task_name:s}...'.format(task_name=task))  
  fid_value_wrt_target_train_set = calculate_fid_given_paths(paths=[real_target_train_path, fake_source2target_path], batch_size=opts.fid_batch_size, 
                                                             device=device, dims=2048, num_workers=1)
  fid_value_wrt_target_test_set = calculate_fid_given_paths(paths=[real_target_test_path, fake_source2target_path], batch_size=opts.fid_batch_size, 
                                                            device=device, dims=2048, num_workers=1)
  
  print('FID with regard to the target train set for {task_name:s} is: {fid_val:.6f}'.format(task_name=task, fid_val=fid_value_wrt_target_train_set))
  print('FID with regard to the target test set for {task_name:s} is: {fid_val:.6f}'.format(task_name=task, fid_val=fid_value_wrt_target_test_set))
  return fid_value_wrt_target_train_set, fid_value_wrt_target_test_set


def compute_prdc_two_domains_unpaired(opts): 
  if opts.a2b: 
    source_domain = 'A'
    target_domain = 'B'
    direction = 'A2B'
  else: 
    print("Need to be set opts.a2b to be 1")
    input()  
    
  # pdb.set_trace()
  real_source_train_path = os.path.join(opts.dataroot, 'train' + source_domain)
  real_target_train_path = os.path.join(opts.dataroot, 'train' + target_domain)
  real_source_test_path = os.path.join(opts.dataroot, 'test' + source_domain )
  real_target_test_path = os.path.join(opts.dataroot, 'test' + target_domain)
  
  fake_source2target_path = os.path.join(opts.result_dir, opts.name, 'fake_{direction:s}'.format(direction=direction))
  task = "{task_name:s}-{source_domain:s}2{target_domain:s}".format(task_name=opts.name, source_domain=source_domain, target_domain=target_domain)
  iprdc = PRDC(opts.prdc_batch_size, opts.k, opts.prdc_num_samples)
  
  # compute Precision & Recall, and Density & Coverage with regard to the train set
  path_real = real_target_train_path
  path_fake = fake_source2target_path
  real_features = iprdc.extract_features_from_files(path_or_fnames=path_real)
  fake_features = iprdc.extract_features_from_files(path_or_fnames=path_fake)
  print('Computing Precision & Recall, Density & Coverage for {task_name:s} with regard to train set'.format(task_name=task))
  # pdb.set_trace()
  with torch.no_grad(): 
    prdc_dict_wrt_target_train_set = iprdc.compute_prdc(real_features=real_features, fake_features=fake_features, nearest_k=opts.nearest_k)
  print('Precision:', prdc_dict_wrt_target_train_set['precision'])
  print('Recall:', prdc_dict_wrt_target_train_set['recall'])
  print('Density:', prdc_dict_wrt_target_train_set['density'])
  print('Coverage:', prdc_dict_wrt_target_train_set['coverage'])
  
  # compute Precision & Recall, and Density & Coverage with regard to the test set
  path_real = real_target_test_path
  path_fake = fake_source2target_path
  real_features = iprdc.extract_features_from_files(path_or_fnames=path_real)
  fake_features = iprdc.extract_features_from_files(path_or_fnames=path_fake)
  print('Computing Precision & Recall, Density & Coverage for {task_name:s} with regard to train set'.format(task_name=task))
  with torch.no_grad(): 
    prdc_dict_wrt_target_test_set = iprdc.compute_prdc(real_features=real_features, fake_features=fake_features, nearest_k=opts.nearest_k)
  print('Precision:', prdc_dict_wrt_target_test_set['precision'])
  print('Recall:', prdc_dict_wrt_target_test_set['recall'])
  print('Density:', prdc_dict_wrt_target_test_set['density'])
  print('Coverage:', prdc_dict_wrt_target_test_set['coverage'])
  
  return prdc_dict_wrt_target_train_set, prdc_dict_wrt_target_test_set  


def main(): 
  # parse options
  parser = TestOptions()
  opts = parser.parse()
  opts.nThreads = 1
  
  # data loader 
  print('\n--- load dataset ---')
  if opts.a2b:
    dataset = dataset_single(opts=opts, setname='A', input_dim=opts.input_dim_a)
  else: 
    print("Need to be set opts.a2b to be 1")
    input()  
    
  loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=opts.nThreads)  
  # model
  print('\n --- load model ---')
  model = DiversityAugmentDRIT(opts=opts)  
  model.setgpu(gpu=opts.gpu)
  model.resume(model_dir=opts.resume, train=False)
  model.eval()
  
  # directory
  result_dir = os.path.join(opts.result_dir, opts.name)
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)
  
  """ evaluation stage  """  
  #  Generate images and Calculate LPIPS
  print('\nGenerating images and calculating LPIPS for {task:s}...'.format(task=opts.name))
  lpips_values = []    
  for idx1, data in enumerate(loader): 
    img1, img_name1 = data[0], data[1][0]
    img1 = img1.cuda()
    image_name = os.path.basename(img_name1)[:-4]
    real_test_images = [img1]
    real_test_image_names = [image_name]
    if opts.a2b: 
      real_image_dir = 'real_A' 
      fake_images_dir = 'fake_{direction:s}'.format(direction='A2B')
    else: 
      print("Need to be set opts.a2b to be 1!")
      input() 
    fake_images = list()
    fake_image_names = list()   
    for idx2 in range(1, opts.num+1): 
      with torch.no_grad(): 
        fake_img = model.test_forward(image=img1, a2b=True) 
        fake_images.append(fake_img)   
        if opts.a2b: 
          fake_image_name = image_name + '_{idx:02d}'.format(idx=idx2)
          fake_image_names.append(fake_image_name)
        else: 
          print("Need to be set opts.a2b to be 1!")
          input()   
    # pdb.set_trace()  
    save_images(images=real_test_images, names=real_test_image_names, path=os.path.join(result_dir, real_image_dir))      
    save_images(images=fake_images, names=fake_image_names, path=os.path.join(result_dir, fake_images_dir))  
    lpips_value = calculate_lpips_given_images(group_of_images=fake_images)  
    print('Calculate LPIPS for the image {:d} / {:d} is {:f}'.format(idx1, len(loader), lpips_value))
    lpips_values.append(lpips_value)
  lpips_mean = np.array(lpips_values).mean()     
  pdb.set_trace()
  print('LPIPS for {task_name:s} is: {lpips_val:.6f}'.format(task_name=opts.name, lpips_val=lpips_mean.astype(np.float)))   
  
  # Calculate FID
  fid_value_wrt_target_train_set, fid_value_wrt_target_test_set = calculate_fid_two_domains_unpaired(opts=opts)    
  # Calculate Precial & Recall, and Density & Coverage
  prdc_dict_wrt_target_train_set, prdc_dict_wrt_target_test_set  = compute_prdc_two_domains_unpaired(opts=opts)
  
  # output
  output_metrics_filename = os.path.join(opts.result_dir, opts.name, 'metrics_{task_name:s}.txt'.format(task_name=opts.name))
  with open(output_metrics_filename, "w") as f: 
    f.write("Task: {task_name:s} \t \n".format(task_name=opts.name))
    f.write("LPIPS: \t{:.6f} \n".format(lpips_mean.astype(np.float)))
    f.write(" with regard to the Train Target Set: \t \n")  
    f.write("FID: \t{:.6f} \n".format(fid_value_wrt_target_train_set))
    f.write("Precision: \t{:.6f} \n".format(prdc_dict_wrt_target_train_set['precision'].astype(np.float)))
    f.write("Recall: \t{:.6f} \n".format(prdc_dict_wrt_target_train_set['recall'].astype(np.float)))
    f.write("Density: \t{:.6f} \n".format(prdc_dict_wrt_target_train_set['density'].astype(np.float)))
    f.write("Coverage: \t{:.6f} \n".format(prdc_dict_wrt_target_train_set['coverage'].astype(np.float)))
    f.write(" with regard to the Test Target Set: \t \n")  
    f.write("FID: \t{:.6f} \n".format(fid_value_wrt_target_test_set))
    f.write("Precision: \t{:.6f} \n".format(prdc_dict_wrt_target_test_set['precision'].astype(np.float)))
    f.write("Recall: \t{:.6f} \n".format(prdc_dict_wrt_target_test_set['recall'].astype(np.float)))
    f.write("Density: \t{:.6f} \n".format(prdc_dict_wrt_target_test_set['density'].astype(np.float)))
    f.write("Coverage: \t{:.6f} \n".format(prdc_dict_wrt_target_test_set['coverage'].astype(np.float)))



if __name__ == '__main__':
  main()
    
  
    