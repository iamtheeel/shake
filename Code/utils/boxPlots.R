# Get the running dir
this_dir <- dirname(sys.frame(1)$ofile)  # If running via source()

#source(file.path(this_dir, "last10Epo_full.R"))
#source(file.path(this_dir, "last10Epo_full_fixedFS.R"))
source(file.path(this_dir, "last10Epo_ds-4x.R"))
#source(file.path(this_dir, "last10Epo_ds-4x_beanStuff.R"))



data_list_mNet <- list(
  titleStr = "Validation for Last 10 Epochs of Each Wavelet Tested, Down Sample = 4x\n MobilNet V2, Batch Size 1, 100 Epochs LR:0.00005 WD:0.0005",
##",
yMin =0.00, 
    yMax = 0.035,
  data = list(
    Time         = data_time,
    Spectra      = data_sGram,
    Ricker       = data_ricker,
    
    `Morlet\nReal`   = data_morl,
    `cMor\nbw:2\nMag` = data_cMorl_morMatch,
    #`cMor\nbw:2\nCplx` = `mNet_cMore-2-cplx_LR0.00005_WD_0.0005_adam_gNorm_bs1_2`,
    
    `cMor\nbw:10\nMag` = data_cMorl,
    #`cMor\nbw:10\nCplx` = `mNet_cMore-10-cplx_LR0.00005_WD_0.0005_adam_gNorm_bs1`,
    
    `Cust\nf0:2.14\nReal`  = data_fStep,
    `cCust\nf0:2.14\nMag` = data_cFStep,
    #`cCust\nf0:2.14\nCplx` = `mNet_cFStep-2.14-cplx_LR0.00005_WD_0.0005_adam_gNorm_bs1`,
    
    
    `Cust\nf0:10\nReal`    = data_fStep_10,
    `cCust\nf0:10\nMag` =  data_cFStep
    #`cCust\nf0:10\nCplx` = `mNet_cFStep-10-cplx_LR0.00005_WD_0.0005_adam_gNorm_bs1`
  )
)

data_list_mNet_LRSCH_bs8_adamW_bNorm <- list(
  titleStr = "Validation for Last 10 Epochs of Each Wavelet Tested, Down Sample = 4x\n MobilNet V2, 225 Epochs LR:0.00005 WD:0.00002 CosAnWarmRes T0:25, T-m:1, eta_min: 1e-7 AdamW Batch Size 8",
  #
  yMin =0.00, 
  yMax = 0.035,
  data = list(
    Time                  = `mNet_timeD-real.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_bNorm_bs8`,
    Spectra               = `mNet_spectra-mag.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_bNorm_bs8`,
    Ricker                = `mNet_ricker-real.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_bNorm_bs8`,
    Morlet                = `mNet_more-real_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_bNorm_bs8`,
    `cMor bw2\n Mag`      = `mNet_cMore-2-mag_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_bNorm_bs8`,
    `cMor bw10\n Mag`     = `mNet_cMore-10-mag_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_bNorm_bs8`,
    `Cust\nf0:2.14`       = `mNet_fStep-2.14-real_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_bNorm_bs8`,
    `Cust\nf0:10`         = `mNet_fStep-10-real_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_bNorm_bs8`,
    `cCust\nf0:2.14\nMag` = `mNet_cFStep-2.14-real_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_bNorm_bs8`
    #`cCust\nf0:10\nMag`   = `mNet_cFStep-10-real_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7`
    )
)

#batch size 1, batch norm did not converge
data_list_mNet_LRSCH_bs1_adamW_gNorm <- list(
  titleStr = "Validation for Last 10 Epochs of Each Wavelet Tested, Down Sample = 4x\n MobilNet V2, 225 Epochs CosAnWarmRes T0:25, AdamW, Batch Size 1, Group Norm",
  #
  yMin =0.00, 
  yMax = 0.035,
  data = list(
    Time                  = `mNet_timeD-real.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_gNorm_bs1`,
    Spectra               = `mNet_spectra-mag.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_gNorm_bs1`,
    Ricker                = `mNet_ricker-real.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_gNorm_bs1`,
    Morlet                = `mNet_more-real_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_gNorm_bs1`,
    `cMor bw2\n Mag`      = `mNet_cMore-2-mag_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_gNorm_bs1`,
    `cMor bw10\n Mag`     = `mNet_cMore-10-mag_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_gNorm_bs1`,
    `Cust\nf0:2.14`       = `mNet_fStep-2.14-real_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_gNorm_bs1`,
    `Cust\nf0:10`         = `mNet_fStep-10-real_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_gNorm_bs1`,
    `cCust\nf0:2.14\nMag` = `mNet_cFStep-2.14-real_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_gNorm_bs1`
    #`cCust\nf0:10\nMag`   = `mNet_cFStep-10-real_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7`
  )
)

data_list_mNet_LRSCH_WTF <- list(
  titleStr = "MobilNet CosAnWarmRes T0:25, cMor bw10",
  yMin =0.00, 
  yMax = (ym <- 0.015), # keep it in a var so we can use it in the list
  data = list(`Original Best` = data_cMorl,
              `LR Sched\nB-Norm\nBS 8\nAdamW`     = `mNet_cMore-10-mag_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_bNorm_bs8`,
              #`LR Sched\nB-Norm\nBS 1\nAdamW`     = c(ym), #Did not converge
              `LR Sched\nG-Norm\nBS 1\nAdamW`     = `mNet_cMore-10-mag_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_gNorm_bs1`,
              #`Fixed LR\nG-Norm\nBS 1\nAdamW`     = `mNet_cMore-10-mag_LR0.00005_WD_0.0005_AdamW_gNorm`,
              #`Fixed LR\nG-Norm\nBS 1\nAdam\n120Epochs`     = `mNet_cMore-10-mag_LR0.00005_WD_0.0005_Adam_gNorm_120epo`,
              #`Fixed LR\nG-Norm\nBS 1\nAdam`     = `mNet_cMore-10-mag_LR0.00005_WD_0.0005_Adam_gNorm`,
              `LR Sched\nG-Norm\nBS 1\nAdam\nWD 0.0005` = `mNet_cMore-10-mag_LR0.00005_WD_0.0005_cosAnnWarRes-T025_Tmul1_etaMin1e-7_gNorm_bs1_adam`,
              `LR Sched\nG-Norm\nBS 1\nAdam\nWD 0.00002` = `mNet_cMore-10-mag_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_gNorm_bs1_adam`,
              
              `LR Sched\nG-Norm\nBS 8\nAdam\nWD 0.0005` = `mNet_cMore-10-mag_LR0.00005_WD_0.0005_cosAnnWarRes-T025_Tmul1_etaMin1e-7_gNorm_bs8_adam`,
              `LR Sched\nB-Norm\nBS 8\nAdam\nWD 0.0005` = `mNet_cMore-10-mag_LR0.00005_WD_0.0005_cosAnnWarRes-T025_Tmul1_etaMin1e-7_bNorm_bs8_adam`
              
  )
)
data_list_mNet_LRSCH_adam <- list(
  titleStr = "MobilNet CosAnWarmRes T0:25, adam, LR: 0.00005, wd:0.0005, bs-1, GroupNorm",
  yMin =0.00, 
  yMax = (ym <- 0.035),# keep it in a var so we can use it in the list
  data = list(#`Original\nBest` = data_cMorl,
              `Time\nDomain` = `mNet_timeD-real.00005_WD_0.0005_cosAnnWarRes-T025_Tmul1_etaMin1e-7_adam_gNorm_bs1`,
              `Spectragram` = `mNet_spectra-mag.00005_WD_0.0005_cosAnnWarRes-T025_Tmul1_etaMin1e-7_adam_gNorm_bs1`,
              # slurm-15798.out   20251026-145452, exp-3
              `Ricker` = `mNet_ricker-real.00005_WD_0.0005_cosAnnWarRes-T025_Tmul1_etaMin1e-7_adam_gNorm_bs1`,
              # slurm-15798.out   20251026-145452, exp-4
              `Morlet` =  `mNet_more-real_LR0.00005_WD_0.0005_cosAnnWarRes-T025_Tmul1_etaMin1e-7_adam_gNorm_bs1`,
              
              `CMore\nbw:2` = `mNet_cMore-2-mag_LR0.00005_WD_0.0005_cosAnnWarRes-T025_Tmul1_etaMin1e-7_adam_gNorm_bs1`,
              `CMore\nbw:10` = `mNet_cMore-10-mag_LR0.00005_WD_0.0005_cosAnnWarRes-T025_Tmul1_etaMin1e-7_adam_gNorm_bs1`,
              
              `fStep\nf0: 2.14` = `mNet_fStep-2.14-real_LR0.00005_WD_0.0005_cosAnnWarRes-T025_Tmul1_etaMin1e-7_adam_gNorm_bs1`,
              # slurm-15799.out   20251026-145552, exp-2
              `fStep\nf0:10` = `mNet_fStep-10-real_LR0.00005_WD_0.0005_cosAnnWarRes-T025_Tmul1_etaMin1e-7_adam_gNorm_bs1`,
              # slurm-15799.out   20251026-145552, exp-3
              `CFStep\nf0:2.14` = `mNet_cFStep-2.14-mag_LR0.00005_WD_0.0005_cosAnnWarRes-T025_Tmul1_etaMin1e-7_adam_gNorm_bs1`,
              # slurm-15799.out   20251026-145552, exp-4
              `CFStep\nf0:10` = `mNet_cFStep-10-mag_LR0.00005_WD_0.0005_cosAnnWarRes-T025_Tmul1_etaMin1e-7_adam_gNorm_bs1`
              
              
  )
)
  

titleStr <- "Complex LeNet"
data_list_leNet <- list(
  yMin =0.00, 
  yMax = 0.035,
  data = list(
  `Morlet\nMobilNet\nReal`     = data_morl,
  `cMor\nbw:2\nMag`      = `leNet_cMore_bw-2_f0-pt7958_mag`,
  `cMor\nbw:2\nComplex`  = `leNet_cMore_bw-2_f0-pt7958_complex`,
  `cMor\nbw:10\nMag`     = `leNet_cMore_bw-10_f0-pt8_mag`,
  `cMor\nbw:10\nComplex` = `leNet_cMore_bw-10_f0-pt8_complex`)
)

data_list_vgg_mNet <- list(
  titleStr = "VGG vs MobilNet",
  yMin =0.00, 
  yMax = 0.035,
  data = list(
    # MobileNet Batch size 1, nEpochs 100
    # VGG Batch size 8, nEpochs ~225
    
    #`mNet\nRicker`       = data_ricker,
    #`VGG\nRicker` = `vgg_ricker_b-8_norm-g_225epo`,
    
    `mNet\nMorlet\nReal`       = data_morl,
    `mNet\ncMor bw:2\nMag` = data_cMorl_morMatch,
    `mNet\ncMore bw2\nCplx\n bs8` = `mNet_cMore-2-cplx_LR0.00005_WD_0.0005_adam_gNorm_bs8`,
    `mNet\ncMore bw2\nCplx\n bs1` = `mNet_cMore-2-cplx_LR0.00005_WD_0.0005_adam_gNorm_bs1`,
    `mNet\ncMore bw2\nCplx\n bs1 take2` = `mNet_cMore-2-cplx_LR0.00005_WD_0.0005_adam_gNorm_bs1_2`,
    
    #`VGG\ncMore bw2\n Mag` = `vgg_cMore-mag_bw2_b-8_norm-g_225epo`,
    `VGG\nMorlet\nReal` = `vgg_more-real_b-8_norm-g_225epo`,
    `VGG\ncMore bw2\nMag` = `vgg_cMore-2-mag_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7`,
    `VGG\ncMore bw2\nCplx` = `vgg_cMore-2-cplx_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7`
    
    #`mNet\ncMor\nbw:10` = data_cMorl,
    #`VGG\ncMore bw10\n Mag\nBatch 8\n225 epochs` = `vgg_cMore-mag_bw10_b-8_norm-g_225epo`,
    
    #`mNet\nfStep\nf0:2.14`  = data_fStep,
    #`VGG\nfStep f0:2.14\n Real\nBatch 8\n240 Epochs` = `VGG_fStep-real-fz2.14_bs8_ep225`,
    
    #`mNet\nfStep\nf0:10`    = data_fStep_10,
    #`VGG\nfStep f0:10\n Real\nBatch 8\n240 Epochs` = `VGG_fStep-real-fz10_bs8_ep225`,
    
    #`mNet\nfStep\nf0:2.14` = data_cFStep,
    #`VGG\ncFStep bw2.14\n mag\nBatch 8\n225 Epochs` = `VGG_cFStep-mag-fz2.14_bs8_ep225`,
    
    #`mNet\nfStep\nf0:10` = c( data_fStep_10),
    #`VGG\ncFStep bw10\n mag\nBatch 8\n225 Epochs` = `VGG_cFStep-mag-fz10_bs8_ep225`

  )
)

data_list_vgg_mNet_cvCNN <- list(
  titleStr = "Complex Valued NN",
  yMin =0.00, 
  yMax = 0.035,
  data = list(
    # MobileNet Batch size 1, nEpochs 100
    # VGG Batch size 8, nEpochs ~225
    
    #`mNet\nRicker`       = data_ricker,
    #`VGG\nRicker` = `vgg_ricker_b-8_norm-g_225epo`,
    
    `mNet\nMorlet\nReal`       = data_morl,
    `mNet\ncMor bw:2\nMag` = data_cMorl_morMatch,
    
    #`VGG\ncMore bw2\n Mag` = `vgg_cMore-mag_bw2_b-8_norm-g_225epo`,
    `VGG\nMorlet\nReal` = `vgg_more-real_b-8_norm-g_225epo`,
    `VGG\ncMore bw2\nMag` = `vgg_cMore-2-mag_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7`,
    `VGG\ncMore bw2\nCplx` = `vgg_cMore-2-cplx_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7`

    
  )
)

data_list_vgg <- list(
  titleStr = "VGG - epochs",
  yMin =0.00, 
  yMax = 0.035,
  data = list(
    #`M Net Morlet\nReal`     = data_morl,
    #`VGG PreMod\nmore\nreal 50 epo` = `VGG-firstLight_more_50eop`,
    #`VGG cMore\n Magnitude 50 epo` = `VGG-_cMore-mag_bw-2_f0-pt8_50eop`,
    #`VGG cMore\n Complex 50 epo` = `VGG-_cMore-complex_bw-2_f0-pt8_50eop`,
    `Ricker\nReal\nB8` = `vgg_ricker_b-8_norm-g_225epo`,
    `More\nReal\nB8` = `vgg_more-real_b-8_norm-g_225epo`,
    `fStep f0:2.14\n Real\nBatch 8\n240 Epochs` = `VGG_fStep-real-fz2.14_bs8_ep225`,
    `fStep f0:10\n Real\nBatch 8\n240 Epochs` = `VGG_fStep-real-fz10_bs8_ep225`,
    
    #`More b=1\n Real\n100 epo` = `VGG-_more`,
    #`cMore bw2 b= 1\n Magnitude\n100 epo` = `VGG-_cMore-mag_bw-2_f0-pt8`,
    `cMore bw2\n Mag\nBatch 8\n225 epochs` = `vgg_cMore-mag_bw2_b-8_norm-g_225epo`,
    `cMore bw10\n Mag\nBatch 8\n225 epochs` = `vgg_cMore-mag_bw10_b-8_norm-g_225epo`,
    `cFStep bw2.14\n mag\nBatch 8\n225 Epochs` = `VGG_cFStep-mag-fz2.14_bs8_ep225`,
    `cFStep bw10\n mag\nBatch 8\n225 Epochs` = `VGG_cFStep-mag-fz10_bs8_ep225`,
    
    `cMore bw2\n Complex\nBatch 8\n240 Epochs` = `VGG_cMore-complex_bw-2_f0-pt8_240eop`,
    `cMore bw10\n Complex\nBatch 8\n225 Epochs` = `VGG_cMore-CPLX-bw10_bs8_ep225`,

    `cFStep f02.14\n Complex\nBatch 8\n178 Epochs` =`VGG_cFStep-complex_fz2.14_178epo`,
    `cFStep f010\n Complex\nBatch 8\n162 Epochs` =`VGG_cFStep-complex_fz10_162epo`,
    
    #`M-Net b=1\ncMor bw2\n` = data_cMorl_morMatch,
    `M-Net b=1\ncMor bw10` = data_cMorl
  )
)

data_list_vgg_batch_norm <- list(
  titleStr = "batch norm vs group norm on VGG",
  yMin =0.00, 
  yMax = 0.035,
  data = list(
  ## where is this `VGG More\n Real\nG-norm\nBatch 1` = `VGG-_more`,
  `VGG cMore\n Mag\nG-norm\nBatch 1` = `VGG-_cMore-mag_bw-2_f0-pt8`,
  `More\n Real\nG-norm` = `vgg_more-real_b-8_norm-g`,
  `cMore bw:2\nMag\nG-norm` = `vgg_cMore-mag_b-8_norm-g`,
  `cMore bw:2\nComplex\nG-norm` = `vgg_cMore-complex_b-8_norm-g`,
  
  `More\n Real\nB-norm` = `vgg_more-real_b-8_norm-b`,
  `cMore bw:2\nMag\nB-norm` = `vgg_cMore-mag_b-8_norm-b`,
  `cMore bw:2\nComplex\nB-norm` = `vgg_cMore-complex_b-8_norm-b`)
  
)

data_list_mNet_batchSize <- list(
  titleStr = "Batch size 1,8, 16: Mobilnet",
  yMin =0.00, 
  yMax = 0.03,
  data = list(
    `Ricker\nBatchSize 1`       = data_ricker,
    `Morlet Real\nBatchSize 1`  = data_morl,
    `cMor:2 mag\nBatchSize 1`   = data_cMorl_morMatch,
    
    `Ricker\nBatchSize 8`       = data_ricker_mNet_bs8,
    `Morlet Real\nBatchSize 8`  = `data_morl-real_mNet_bs8`,
    `cMor:2 Mag\nBatchSize 8`   = `data_cmorl-bw2-mag_mNet_bs8`,
    
    `Ricker\nBatchSize 16`      = `data_ricker_mNet_bs16`,
    `Morlet Real\nBatchSize 16` =`data_morl-real_mNet_bs16`,
    `cMor:2 Mag\nBatchSize 16`  = `mNet-cMorel-bw2_mag_bs16`
    )
)

data_list_mNet_bSvLR <- list(
  titleStr = "Batch size 1 and 8, LR scheduler on Mobilnet",
  yMin =0.00, 
  yMax = 0.03,
  data = list(
    `Ricker\nBatchSize 1`      = data_ricker,
    `Morlet Real\nBatchSize 1` = data_morl,
    `cMor:2 mag\nBatchSize 1`  = `data_cMorl_morMatch`,
    
    `Ricker\nBatchSize 8`       = data_ricker_mNet_bs8,
    `Morlet Real\nBatchSize 8`  = `data_morl-real_mNet_bs8`,
    `cMor:2 Mag\nBatchSize 8`   = `data_cmorl-bw2-mag_mNet_bs8`,
    
    `Ricker\nBatchSize 8\nLR Sched`     = `mNet_ricker-real.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_bNorm_bs8`,
    `Morlet Real\nBatchSize 8\nLR Sched`  = `mNet_more-real_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_bNorm_bs8`,
    `cMor:2 Mag\nBatchSize 8\nLR Sched`   = `mNet_cMore-2-mag_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7_bNorm_bs8`
  )
)

data_list_VGG_batchSize <- list(
  titleStr = "Batch size 1,8: VGG",
  yMin =0.00, 
  yMax = 0.03,
  data = list(
    #`VGG More\n Real\nBatch 1` = `VGG_more`,
    #`VGG cMore\n Mag\nBatch 1` = `VGG_cMore-mag_bw-2_f0-pt8`,
    
    #`VGG More\n Real\nBatch 8\n100 epochs` = `vgg_more-real_b-8_norm-g`,
    #`VGG More\n Mag\nBatch 8\n100 epochs` = `vgg_cMore-mag_b-8_norm-g`,
    
    `VGG More\n Real\nBatch 8\n225 epochs` = `vgg_more-real_b-8_norm-g_225epo`
    #`VGG More\n Mag\nBatch 8\n225 epochs` = `vgg_cMorebw-mag_b-8_norm-g_225epo`

  )
)


# 3x is slightly better than 4x
data_list_ds <- list(
  yMin =0.00, 
  yMax = 0.035,
  #titleStr <- "Comparison of downscaled runs on Mobilnet with Complex Morlet (bw:10, f0:0.8) CWT"
  titleStr <- "",
  data = list(`1x` = `ds_1x`,`4x` = `ds_4x`)
  #`1x` = `ds_1x`, `2x` = `ds_2x`,`3x` = `ds_3x`,`4x` = `ds_4x`
)

data_list_vgg_complex <- list(
  titleStr = "VGG LR:0.00005 WD:0.0005, batch size: 8",
  yMin =0.00, 
  yMax = 0.03,
  data = list(
    #`More\n Real\nBatch 8\n100 epochs` = `vgg_more-real_b-8_norm-g`,
    #`cMore\n Mag\nBatch 8\n100 epochs` = `vgg_cMore-mag_b-8_norm-g`,
    #`More\n Real\nBatch 8\n250 epochs` = `vgg_more-real_b-8_norm-g_225epo`,
    `cMore bw2\n Mag` = `VGG-_cMore-mag_bw-2_f0-pt8`, #vgg_cMorebw-mag_b-8_norm-g_225epo`,
    #`cMore\n Complex\nBatch 1\n50 Epochs` = `VGG-_cMore-complex_bw-2_f0-pt8_50eop`,
    `cMore bw2 Cplx` = `VGG_cMore-complex_bw-2_f0-pt8_240eop`,
    `cMore bw 10 Cplx` = `VGG_cMore-CPLX-bw10_bs8_ep225`,
    `cFStep f0 2.14 Cplx` = `VGG_cFStep-complex_fz2.14_178epo`,
    `cFStep f0 10Cplx` = `VGG_cFStep-complex_fz10_162epo`
    #`M-Net\ncMor bw2` = data_cMorl_morMatch,
    #`M-net\ncMor bw10` = data_cMorl
  )
)

data_list_vgg_LR <- list(
  titleStr = "VGG LR:0.00005 WD:0.00002 CosAnWarmRes T0:25, T-m:1, eta_min: 1e-7 AdamW",
  # batch size 8
  yMin =0.00, 
  yMax = 0.035,
  data = list(
    #`cMore bw2 Mag\nLR:0.00005 WD:0.0005\nAdam` = `vgg_cMore-mag_bw2_b-8_norm-g_225epo`,
    #`cMore bw2 Cplx\nLR:0.00005 WD:0.0005\nAdam` = `VGG_cMore-complex_bw-2_f0-pt8_240eop`,
    #`cMore bw2 Mag` = `vgg_cMore-2-mag_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7`,
    
    `cMore bw2 Cplx` = `vgg_cMore-2-cplx_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7`,
    `cMore bw 10 Cplx` = `vgg_cMore-10-cplx_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7`,
    `cFStep f0 2.14 Cplx` = `vgg_fStep-2.14_cplx_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7`,
    `cFStep f0 10Cplx` = `vgg_fStep-10-cplx_LR0.00005_WD_0.00002_cosAnnWarRes-T025_Tmul1_etaMin1e-7`,
    
    `M-Net Best\ncMor bw10` = data_cMorl
  )
)

dataList = data_list_mNet

#dataList = data_list_mNet_LRSCH
#dataList = data_list_mNet_bSvLR
#dataList = data_list_mNet_batchSize
#dataList = data_list_mNet_LRSCH_bs1_adamW_gNorm
#dataList = data_list_mNet_LRSCH_WTF
#dataList = data_list_mNet_LRSCH_adam

#dataList = data_list_vgg_mNet

#dataList = data_list_vgg_mNet_cvCNN

#dataList = data_list_vgg_batch_norm
#dataList = data_list_VGG_batchSize
#dataList = data_list_vgg_complex
#dataList = data_list_vgg
#dataList = data_list_vgg_LR


# Convert the data to data.frame with columns: values, group
df <- stack(dataList$data)

ymin <- dataList$yMin
ymax <- dataList$yMax


boxplot(values ~ ind, data = df,
      main = "", xlab = "", ylab = "Validation Error (m/s rms)",
      horizontal = FALSE, ylim = c(ymin, ymax),
      axes = FALSE)

title(dataList$titleStr, line = 1)
nTickValues <- 9
axis(2, at = c(ymin, pretty(df$values, n=nTickValues), ymax)) 
grid(nx = NA, ny = NULL, col = "lightgray", lty = "dotted", lwd = 1)
axis(1, at = seq_along(levels(df$ind)), labels = levels(df$ind))
box() # Put the box back


# Do some stats
summary_table <- aggregate(values ~ ind, data = df, FUN = function(x) {
  c(min = min(x),
    max = max(x),
    mean = mean(x),
    median = median(x),
    sd = sd(x)) #in R, the default is sample std
})

# expand the matrix column into separate columns
summary_table <- do.call(data.frame, summary_table)
print(summary_table)


## which is better
anova_model <- aov(values ~ ind, data = df)
ano <- summary(anova_model)
print(ano)

# Post-hoc comparisons (Tukey HSD)
# Tukey HSD
tuk <- TukeyHSD(anova_model)
print(tuk)                    # <-- explicit print


pairwise_results <- pairwise.t.test(df$values, df$ind,
                                    p.adjust.method = "bonferroni")
print(pairwise_results)
