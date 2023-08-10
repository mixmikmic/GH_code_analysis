get_ipython().magic('pylab inline')
import pysd
import numpy as np

model = pysd.read_vensim('../../models/Pendulum/Single_Pendulum.mdl')

angular_position = np.linspace(-1.5*np.pi, 1.5*np.pi, 60)
angular_velocity = np.linspace(-2, 2, 20)

apv, avv = np.meshgrid(angular_position, angular_velocity)

def derivatives(ap, av):
    ret = model.run(params={'angular_position':ap,
                            'angular_velocity':av}, 
                    return_timestamps=[0,1],
                    return_columns=['change_in_angular_position',
                                    'change_in_angular_velocity'])

    return tuple(ret.loc[0].values)

derivatives(0,1)

vderivatives = np.vectorize(derivatives)

dapv, davv = vderivatives(apv, avv)
(dapv == avv).all()

plt.figure(figsize=(18,6))
plt.quiver(apv, avv, dapv, davv, color='b', alpha=.75)
plt.box('off')
plt.xlim(-1.6*np.pi, 1.6*np.pi)
plt.xlabel('Radians', fontsize=14)
plt.ylabel('Radians/Second', fontsize=14)
plt.title('Phase portrait for a simple pendulum', fontsize=16);

