# SRS-map2map
A trained super-resolution model for N-body cosmological simulation


## Model

SR model super-resolves the N-body cosmological simulation. It generates 512 times more tracer particles and produces their full phase-space (displacment+velocity) distribution.

SRmodel/G_z0.pt and G_z2.pt are the trained SR models to super-resolve the N-body cosmological simulation at z=2 and z=0 separately.  
They are the models used in [PaperII](https://doi.org/10.1093/mnras/stab2113)

For details of the model archetecture and training, check repository [map2map](https://github.com/eelregit/map2map), also [PaperI](https://www.pnas.org/content/118/19/e2022038118) and [PaperII](https://doi.org/10.1093/mnras/stab2113).


## Usage

We use [MP-Gadget](https://github.com/MP-Gadget/MP-Gadget) to run cosmological simulation for the LR input. The input and output snapshots are stored in [bigfile](https://github.com/rainwoodman/bigfile) format.

**Step 1** : Run a low-resolution N-body simulation. Our model is trained on LR sets with {Boxsize=100 Mpc/h, Ng_lr=64}. The test LR simulation should in the same resolution (e.g., Ng=128 with Boxsize=200 Mpc/h). In `scripts/LR-sim`, we provide the parameter files used to run LR simulation with [MP-Gadget](https://github.com/MP-Gadget/MP-Gadget).


**Step 2** : `preproc.py` convert the snapshot of N-body simulation to a 3D image with 6 channels, in shape of `(Nc,Ng,Ng,Ng)`. Here `Nc=6` are the normalized displacement + velocity field of tracer particles arranged by their original grid. For usage, check `scripts/preproc.slurm` as an example job script. 


**Step 3** : `lr2sr.py` use the trained SR mdoel to super-resolve the LR input and output the `Position` and `Velocity` column of 512 more tracer particles as the SR output. For usage, check `scripts/lr2sr.py` as an example job script. 


## Note

`lr2sr.py` assumes you use a GPU node. If you use a CPU node instead (which would be much slower), you might need to change `lr2sr.py#30` to `device = torch.device('cpu')`.

Because the limitation of GPU memory, we chunk the LR input to pieces of `nsplit^3` for super resolution. You can set `nsplit` argument larger if output of GPU memory, but make sure that nsplit devides Ng_lr. 


