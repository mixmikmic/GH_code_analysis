# Double-peak KL loss:
def vae_loss(x, decoded):  
    xent_loss = K.sum(K.sum(objectives.binary_crossentropy(x ,decoded),axis=-1),axis=-1)
    #kl_loss_d1 = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) 
    m = K.constant(0)
    s = K.constant(1)
    # the following line: when m=0,s=1, it should be a single peak normal distribution
    #kl_loss_d1 = K.sum(K.log(2/K.exp(z_log_var/2))+(K.square(z_mean)+(K.exp(z_log_var/2)-K.constant(1))*(K.exp(z_log_var/2)+K.constant(1)))/(K.constant(2)),axis = -1)
    kl_loss_d1 = K.sum(K.log(2*s/K.exp(z_log_var/2))+(K.constant(2)*m*(-K.exp(-(K.square(z_mean))/((K.constant(2))*K.exp(z_log_var)))*K.exp(z_log_var/2) + K.sqrt(K.constant(np.pi/2))*z_mean*(K.constant(1)-K.tanh(K.constant(1.19)*z_mean/K.constant(np.sqrt(2))/K.exp(z_log_var/2)))) )/(K.square(s))+(K.square(m-z_mean)+(K.exp(z_log_var/2)-s)*(K.exp(z_log_var/2)+s))/(K.constant(2)*K.square(s)),axis = -1)
    return 1*xent_loss + 1*kl_loss_d1 

### PW loss:
def vae_loss(x, decoded):  
    xent_loss = K.sum(K.sum(objectives.binary_crossentropy(x ,decoded),axis=-1),axis=-1)#the cross entropy loss, can also use MSE
    w_loss_d1_mean = K.sum(K.square((z_mean)), axis=-1) #first part of PW loss
    m = K.constant(0) #hyperparameter: mean
    s = K.constant(1) #hyperparameter: std
    w_loss_d1_var = K.sum((        
        K.exp(-K.square(m/s)-K.square(z_mean)/K.constant(2)/(K.square(s)+K.exp(z_log_var)))*
            (
            -K.constant(2*np.sqrt(2))*K.exp((m*(-K.constant(2)*z_mean*K.square(s)+m*(K.square(s)+K.constant(2)*K.exp(z_log_var))))/(K.constant(2)*K.square(s)*(K.square(s)+K.exp(z_log_var))))
            -K.constant(2*np.sqrt(2))*K.exp((m*(K.constant(2)*z_mean*K.square(s)+m*(K.square(s)+K.constant(2)*K.exp(z_log_var))))/(K.constant(2)*K.square(s)*(K.square(s)+K.exp(z_log_var))))
            +K.exp(z_log_var/2)*K.exp((K.square(z_mean))/(K.constant(2)*(K.square(s)+K.exp(z_log_var))))*K.sqrt(K.constant(1)/K.square(s)+K.constant(1)/K.exp(z_log_var))
            +K.exp(K.square(m/s)+(K.square(z_mean))/(K.constant(2)*(K.square(s)+K.exp(z_log_var))))*K.exp(z_log_var/2)*s*(K.constant(1)/s+K.constant(2)/K.exp(z_log_var/2))*K.sqrt(K.constant(1)/K.square(s)+K.constant(1)/K.exp(z_log_var))
            )
        )
        /(K.constant(4)*K.constant(np.sqrt(np.pi))*s*K.exp(z_log_var/2)*K.sqrt(K.constant(1)/K.square(s)+K.constant(1)/K.exp(z_log_var))), axis=-1)
    #second part of PW loss
    return 1*xent_loss +w_loss_d1_mean+w_loss_d1_var 
    



