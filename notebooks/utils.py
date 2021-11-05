import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import torch
import time


from monai.transforms import KeepLargestConnectedComponent
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.metrics import compute_meandice
from monai.metrics import compute_hausdorff_distance
from monai.metrics import compute_average_surface_distance


from monai.data import write_nifti
from tqdm.notebook import tqdm


def evaluate(model, data_loader, output_path=None, roi_size=(224,224,128), plot_results=False, max_eval=None, filename_prefix_key=None):

    df = pd.DataFrame()

    model.eval()
    device = torch.device("cuda:0")
    model.to(device)

    dice_metric = DiceMetric(include_background=False, to_onehot_y=False, sigmoid=False, reduction="none")

    input_paths = list()
    output_results = list()
    
    original_dice_results = list()
    postprocess_dice_results = list()

    original_hd_results = list()
    postprocess_hd_results = list()

    original_mad_results = list()
    postprocess_mad_results = list()
    
    inference_times = list()


    with torch.no_grad():
        for i, test_data in tqdm(enumerate(data_loader)):

            # Early stopping
            if max_eval is not None:
                if i>=max_eval:
                    break
            
            input_file_name = test_data['IMAGE_meta_dict']['filename_or_obj'][0]
            input_paths.append(input_file_name)
            
            sw_batch_size = 4
            
            inference_data = test_data['IMAGE'].to(device)
            
            start_time = time.time()
            test_outputs = sliding_window_inference(inference_data, roi_size, sw_batch_size, model)
            inference_time = time.time() - start_time
            
            inference_times.append(inference_time)
            
            test_labels = test_data['SEGM'].to(device)

            argmax = torch.argmax(test_outputs, dim=1, keepdim=True)

            value = dice_metric(y_pred=argmax, y=test_labels)
#             print('Dice: {:.5f}'.format(value.item()))
            original_dice_results.append(value.item())

            hd_value = compute_hausdorff_distance(argmax, test_labels, label_idx=1, percentile=95)
#             print('HD95: {:.5f}'.format(hd_value.item()))
            original_hd_results.append(hd_value)

            mad_value = compute_average_surface_distance(argmax, test_labels, label_idx=1)
#             print('MAD: {:.5f}'.format(mad_value.item()))
            original_mad_results.append(mad_value)

            # Post process results
            largest = KeepLargestConnectedComponent(applied_labels=[1])(argmax)
            value = dice_metric(y_pred=largest, y=test_labels)
            postprocess_dice_results.append(value.item())
#             print('Post-processed Dice: {:.5f}'.format(value.item()))

            hd_value = compute_hausdorff_distance(largest, test_labels, label_idx=1, percentile=95)
#             print('Post-processed HD95: {:.5f}'.format(hd_value.item()))
            postprocess_hd_results.append(hd_value)

            mad_value = compute_average_surface_distance(largest, test_labels, label_idx=1)
#             print('Post-processed HD95: {:.5f}'.format(mad_value.item()))
            postprocess_mad_results.append(mad_value)

            if plot_results:
                slice_num = test_data['IMAGE'].shape[-1]//2

                plt.figure('check', (18, 6))
                plt.subplot(1, 3, 1)
                plt.title('image ' + str(i))
                plt.imshow(test_data['IMAGE'][0, 0, :, :, slice_num], cmap='gray')
                plt.subplot(1, 3, 2)
                plt.title('label ' + str(i))
                plt.imshow(test_data['SEGM'][0, 0, :, :, slice_num])
                plt.subplot(1, 3, 3)
                plt.title('output %d' % (i))
                plt.imshow(largest.detach().cpu()[0, 0, :, :, slice_num])
                plt.show()

            if output_path is not None:
                # Get the image affine matrix
                current_affine = test_data['IMAGE_meta_dict']['affine'][0].numpy()
                original_affine = test_data['IMAGE_meta_dict']['original_affine'][0].numpy()
                original_spatial_shape = test_data['IMAGE_meta_dict']['spatial_shape'][0].numpy()

                # Write data out
                input_file_name = test_data['IMAGE_meta_dict']['filename_or_obj'][0]
                output_file_name = os.path.split(input_file_name)[1]
                output_root_name = output_file_name[:-len('.nii.gz')]
                
                
                prefix = ''
                
                if filename_prefix_key:
                    prefix = str(test_data[filename_prefix_key][0]) + '_'
                    
                
                
                output_path_nifti = os.path.join(output_path,'{}{}_segm.nii.gz'.format(prefix, output_root_name))
                print(output_path_nifti)
                output_results.append(output_path_nifti)

                write_nifti(argmax.cpu()[0, 0,...].numpy(),
                            output_path_nifti,
                            mode='nearest',
                            affine=current_affine, 
                            target_affine=original_affine,
                            output_spatial_shape=original_spatial_shape, 
                            dtype=np.float32
                           )

            
    df['DATA_PATH'] = input_paths
    df['DICE'] = original_dice_results
    df['POST_DICE'] = postprocess_dice_results
    df['HD95'] = original_hd_results
    df['POST_HD95'] = postprocess_hd_results
    df['MAD'] = original_mad_results
    df['POST_MAD'] = postprocess_mad_results
    df['InferenceTime'] = postprocess_mad_results
    if output_path is not None:
        df['RESULT_PATH'] = output_results
            
    return df