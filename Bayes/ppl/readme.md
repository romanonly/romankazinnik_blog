Blog post: https://www.romankazinnik.com/feed/beyond-supervised-machine-learning-learning-latent-distributions

Supervised ML can be a powerful tool, however there is a range of problems where its utilization can't be justified. Such problems include reasoning, distributions learning. U til recently solving large-scale problems was limited to Supervised ML. Introduction of deep-learning PyTorch based pyro.ai opens opportunities to apply Unsupervised Learning to large-scale problems.

=============
Banner data
=============

1. data_read_banner.Rmd	
read and plot banner data 

![Alt text](https://user-images.githubusercontent.com/17115347/52920565-9a94d480-32c2-11e9-95c6-394b63094abe.png?raw=true "Banner data")

2. train_webppl_models.Rmd	
train two models

3. plot_webppl_models.Rmd	
plot resulting distributions for two models

![Alt text](https://user-images.githubusercontent.com/17115347/52920359-5a345700-32c0-11e9-942b-4c83e99280ad.png?raw=true "Model-2 (top) and Model-1")

![Alt text](https://user-images.githubusercontent.com/17115347/52920361-5b658400-32c0-11e9-9736-ba5578807461.png?raw=true "Time histogram (top) and Time pdf-s for Robot (green) and Human (red)")


4. plot_webppl_models_shiny.Rmd	
interactive plots of distributions for two models

=============
Countries data: https://github.com/nirupamaprv/Analyze-AB-test-Results 
=============
countries.csv
ab_edited.csv


1.1 Countries original data (200Km samples)
![Alt text](https://user-images.githubusercontent.com/17115347/52920471-9320fb80-32c1-11e9-9031-5305ffce7061.png?raw=true "Countries data")

1.2 Countries small sample data (200 samples)
![Alt text](https://user-images.githubusercontent.com/17115347/52920502-e85d0d00-32c1-11e9-8323-a547462dcb2a.png?raw=true "Countries data")

2. Modeling Robot-Human hypothesis

![Alt text](https://user-images.githubusercontent.com/17115347/52920367-6d472700-32c0-11e9-9833-59ff71bc576b.png?raw=true "Model-2 (top) and Model-1")

3. Hypothesis Robot-Human correctly confirmed by the original data 
![Alt text](https://user-images.githubusercontent.com/17115347/52920470-92886500-32c1-11e9-9ff2-6216959443cf.png?raw=true "Model-2 (top) and Model-1")













