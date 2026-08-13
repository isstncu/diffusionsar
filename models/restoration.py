import torch
import utils
import os

class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, test_loader, r=None):
        image_folder = os.path.join(self.args.image_folder)
        with torch.no_grad():
            #for i, (xx,filename) in enumerate(test_loader):#gaussian noise
            for i, (xx,filename, lambda_) in enumerate(test_loader):#gamma noise
                y = filename[0].split("'")[0]
                print(f"starting processing from image {y}")

                x=xx[0]
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x_cond = xx[0].to(self.diffusion.device)
                xt = xx[1].to(self.diffusion.device)
                
                x_output = self.diffusive_restoration(x_cond, x=xt, r=r)

                #utils.logging.save_image(x_output, os.path.join(image_folder, f"PRED_{y}.tif"))#gaussian noise
                utils.logging.save_image_v2(x_output, lambda_, os.path.join(image_folder, f"PRED_{y}.tif"))#gamma noise
    
    def diffusive_restoration(self, x_cond, x=None, r=None):
        p_size = self.config.data.image_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        
        #x = torch.randn(x_cond.size(), device=self.diffusion.device)#gaussian noise
        x_output = self.diffusion.sample_image(x_cond, x, patch_locs=corners, patch_size=p_size)
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, _, h, w = x_cond.shape
        r = 16 if r is None else r

        h_list = list(range(0, h - output_size + 1, r))
        w_list = list(range(0, w - output_size + 1, r))

        if h_list[-1] != h - output_size:
            h_list.append(h - output_size)

        if w_list[-1] != w - output_size:
            w_list.append(w - output_size)

        return h_list, w_list
        
