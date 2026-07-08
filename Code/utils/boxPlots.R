# Get the running dir
this_dir <- dirname(sys.frame(1)$ofile)  # If running via source()

#source(file.path(this_dir, "last10Epo_full.R"))
#source(file.path(this_dir, "last10Epo_full_fixedFS.R"))
source(file.path(this_dir, "last10Epo_rmCH16.R"))
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

# Split by run, fix stats, rm ch 16, use ch 10 not 8
data_list_RMch16 <- list(
  yMin =0.00, 
  yMax = 0.085,
  titleStr = "Validation Error for Last 10 Epochs",
  
  data = list(
    `Time\n` =`timeD`,
    `Spectra\n` =`sGram`,
    `Ricker\n` =`ricker`,
    `Morlet\n` =`morlet`,
    `Complex\nMorelet\nbw:2` = `cMore-2-cplx`,
    `Complex\nMorelet\nbw:10` = `cMore-10-cplx`,
    `Cust\nf_0:2.14` =`cust-2_14`,
    `Cust\nf_0:10` =`cust-10`,
    `Complex\nCust\nf_0:2.14` =`cCust-2_14`,
    `Complex\nCust\nf_0:10` =`cCust-10`
    )
)


dataList = data_list_RMch16
#dataList = data_list_mNet
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
axis(1, at = seq_along(levels(df$ind)), labels = FALSE) # Shifting, draw the tick marks
axis(1, at = seq_along(levels(df$ind)), labels = levels(df$ind), tick = FALSE,line = 1.5) # then the lables shifted down
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
