get_ipython().magic('matplotlib notebook')

#The following loads the bulk of the hard core code that runs the simulations
#before running this you must install the delta_psi_py package: https://github.com/protonzilla/Delta_Psi_Py
from delta_psi_py import *

#folder to save data:

output_folder='/Users/davidkramer/Dropbox/Data/DF_ECS/'


#Using the standard conditions, set up in the main library, the initial_sim_states is a 
#class that contains the standard initial values. To insert these values into the simulations,
#it is necessary to convert to a list, using the method .as_list(), as in the following:
# initial_sim_states.as_list() 

initial_sim_states=sim_states()
initial_sim_state_list=initial_sim_states.as_list()

Kx_initial=sim_constants()
Kx_initial.k_KEA=0
Kx_initial.fraction_pH_effect=.25
#perform an initial simulation to allow the system to reach equilibrium in the dark


original_dark_equilibrated_initial_y, output=dark_equibration(initial_sim_states.as_list(), 
                                                              Kx_initial, 60*60, return_kinetics=True)
All_Constants_Table('Standard Constants', Kx_initial)

All_Constants_Table('Standard Initial States', initial_sim_states)

# modified_states=sim_states()
# modified_states.Klumen=.02

# print(modified_states.Klumen)

# All_Constants_Table('Standard Initial States', modified_states)

# Changed_Constants_Table('Changed Initial States', initial_sim_states, modified_states)

#generate a dictionary to hold the results of the simulations.

output_dict={}
constants_dict={}
starting_conditions_dict={}

iss=initial_sim_states.as_list()
iss[5]=7.8
iss[6]=0.0

print(iss)

#simulate the effects of a simple turnover flash that only hits PSII centers.
# Alter the kinetic constants so that PSI does not turn over and that the decay of the Dy
#is dominated by the elecrogenic recombination from QA-

#make a new sim_constants, call it K_test_noPSI

K_test_STF=sim_constants() #copy.deepcopy(K)

K_test_STF.DeltaGatp=0

#make mak_b6f =0 so that no b6f turnover
K_test_STF.max_b6f=0

#make kQA=0 as if DCMU was present
K_test_STF.kQA=0

K_test_STF.k_recomb=4*0.33

#slow ATP synthase so that the relaxation of pmf does not interfere with estimates of pmf changes.

K_test_STF.ATP_synthase_max_turnover=0


K_test_STF.Dy=0

K_test_STF.PSI_antenna_size=0


#next generate a short pulse light experiment

max_light_change=1
points_per_segment=100
K_test_STF.perm_K=0


output_dict['single turnover flash_no_counterions'], new_dark_equilibrated_initial_y =sim(K_test_STF, iss, 
        light_pattern['single_turnover_flash'], max_light_change, points_per_segment, dark_equilibration=60*60)

#constants_dict['single turnover flash']=K_test_noPSI
#starting_conditions_dict['single turnover flash']=new_dark_equilibrated_initial_y

K_test_STF.perm_K=100000

output_dict['single turnover flash_val'], new_dark_equilibrated_initial_y =sim(K_test_STF, iss, 
        light_pattern['single_turnover_flash'], max_light_change, points_per_segment, dark_equilibration=60*60)



#plot_interesting_stuff('Figure S1. Test single turnover flash', output)


conditions_to_plot=['single turnover flash_no_counterions', 'single turnover flash_val']
where=[1, 2] #will hold the col and row positions


phenomena_sets=['pmf_params', 'plot_QAm_and_singletO2', 'plot_cum_LEF_singetO2']    
    
        

fig = plt.figure(num='Figure SI 1. Single Turnover Flash', figsize=(5,5), dpi=200) #make a figure
plot_every_nth_point=1
plot_block(output_dict, fig, conditions_to_plot, where, phenomena_sets, plot_every_nth_point)


plt.tight_layout()
plt.show()


plt.savefig(output_folder + 'Figure S1.png', format='png', dpi=200)



#simulate responses to a single square pulse of actinic light
#keeping all rate constants and conditions the same as standard.

on='single square wave permK=normal' #the output name

Kx=sim_constants() #generrate arrays contining optimized time segments for the simulation

Kx.k_KEA=0
Kx.fraction_pH_effect=.25

constants_dict[on]=Kx #store constants in constants_dict

output_dict[on], starting_conditions_dict[on]=sim(Kx, original_dark_equilibrated_initial_y, 
                                        light_pattern['single_square_5_min_300_max'], 
                                        max_light_change, points_per_segment, dark_equilibration=60*60)

Changed_Constants_Table('Change Constants', Kx_initial, Kx)

#plot out the "interesting features"
#plot_interesting_stuff('Test Output 1', output_dict[on])



#simulate responses to a single square pulse of actinic light
#keeping all rate constants and conditions the same as standard.

on='single square wave permK=normal' #the output name

Kx=sim_constants() #generrate arrays contining optimized time segments for the simulation
Kx.k_KEA=0
Kx.fraction_pH_effect=.25
constants_dict[on]=Kx #store constants in constants_dict

output_dict[on], starting_conditions_dict[on]=sim(Kx, original_dark_equilibrated_initial_y, 
                                        light_pattern['single_square_5_min_300_max'], 
                                        max_light_change, points_per_segment, dark_equilibration=60*60)

Changed_Constants_Table('Change Constants', Kx_initial, Kx)

#plot out the "interesting features"
#plot_interesting_stuff('Test Output 2',output_dict[on])



#simulate responses to a single square pulse of actinic light
#keeping all rate constants and conditions the same as standard.

on='single square wave permK=normal' #the output name

Kx=sim_constants() #generrate arrays contining optimized time segments for the simulation

Kx.k_KEA=0
Kx.fraction_pH_effect=.25

constants_dict[on]=Kx #store constants in constants_dict

output_dict[on], starting_conditions_dict[on]=sim(Kx, original_dark_equilibrated_initial_y, 
                                        light_pattern['single_square_5_min_300_max'], 
                                        max_light_change, points_per_segment, dark_equilibration=60*60*4)

Changed_Constants_Table('Change Constants', Kx_initial, Kx)

#plot out the "interesting features"
#plot_interesting_stuff('Test Output 3', output_dict[on])

#simulate responses to a single square pulse of actinic light
#buty with 10-fold more active counterion transporter.


on='single square wave permK=fast' #the output name

Kx=sim_constants() #generrate arrays contining optimized time segments for the simulation
Kx.k_KEA=0
Kx.fraction_pH_effect=.25

Kx.perm_K=10*Kx.perm_K #multiple the perm_K value by 10

constants_dict[on]=Kx #store constants in constants_dict

output_dict[on], starting_conditions_dict[on]=sim(Kx, original_dark_equilibrated_initial_y, 
                                                  light_pattern['single_square_5_min_300_max'], 
           max_light_change, points_per_segment, dark_equilibration=60*60)


#indicate any changes in constants
Changed_Constants_Table('Change Constants', Kx_initial, Kx)
#plot out the "interesting features"
#plot_interesting_stuff('Test Output 4', output_dict[on])


#simulate responses to a single square pulse of actinic light
#buty with 100-fold more active counterion transporter.


on='single square wave permK=very fast' #the output name
Kx=sim_constants() #generrate arrays contining optimized time segments for the simulation
Kx.k_KEA=0
Kx.fraction_pH_effect=.25

Kx.perm_K=100*Kx.perm_K #multiple the perm_K value by 10

constants_dict[on]=Kx #store constants in constants_dict

output_dict[on], starting_conditions_dict[on]=sim(Kx, original_dark_equilibrated_initial_y, 
                                                  light_pattern['single_square_5_min_300_max'], 
           max_light_change, points_per_segment, dark_equilibration=60*60)

#indicate any changes in constants
Changed_Constants_Table('Change Constants', Kx_initial, Kx)

#plot out the "interesting features"
#plot_interesting_stuff('Test Output 5', output_dict[on])



#simulate responses to a single square pulse of actinic light
#buty with NO counterion movements 

on='single square wave permK=0' #the output name


Kx=sim_constants() #generate arrays contining optimized time segments for the simulation
Kx.k_KEA=0
Kx.fraction_pH_effect=.25

Kx.perm_K=0 #

constants_dict[on]=Kx #store constants in constants_dict

output_dict[on], starting_conditions_dict[on]=sim(Kx, original_dark_equilibrated_initial_y, 
                                        light_pattern['single_square_5_min_300_max'], 
                                        max_light_change, points_per_segment, dark_equilibration=60*60*10)
#indicate any changes in constants
Changed_Constants_Table('Change Constants', Kx_initial, Kx)

#plot out the "interesting features"
#plot_interesting_stuff('Test Output 6', output_dict[on])






#set up a list, called conditions_to_plot, that holds the sets of simulation conditions 


conditions_to_plot=['single square wave permK=0', 
                    'single square wave permK=normal', 
                   'single square wave permK=fast']
where=[1,2,3]

#set up a list, called phenomena_sets, that holds the sets of simulation outputs to be 
#plotted.

phenomena_sets=['pmf_params_offset', 
                'b6f_and_balance',
                'K_and_parsing', 
                'plot_QAm_and_singletO2', 
                'plot_cum_LEF_singetO2']    
    
fig = plt.figure(num='Figure 3 (as in paper)', figsize=(5,7), dpi=200) #make a figure
plot_every_nth_point=1 #if = 1, plot all the points

#plot the matrix of graphs

plot_block(output_dict, fig, conditions_to_plot, where, phenomena_sets, plot_every_nth_point)

#make the graphs fit better 
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=.2)

plt.show()

#save the plot
plt.savefig(output_folder + 'compare_K_perm_rates_5_min_square.png', format='png', dpi=200)


#set up a list, called conditions_to_plot, that holds the sets of simulation conditions 


conditions_to_plot=['single square wave permK=0', 
                    'single square wave permK=normal', 
                   'single square wave permK=fast']
where=[1,2,3]

#set up a list, called phenomena_sets, that holds the sets of simulation outputs to be 
#plotted.

phenomena_sets=['pmf_params', 
                'b6f_and_balance',
                'K_and_parsing', 
                'plot_QAm_and_singletO2', 
                'plot_cum_LEF_singetO2']    
    
fig = plt.figure(num='Figure 3-modified (Same as Figure 3, but pmf not offset)', figsize=(5,7), dpi=200) #make a figure
plot_every_nth_point=1 #if = 1, plot all the points

#plot the matrix of graphs

plot_block(output_dict, fig, conditions_to_plot, where, phenomena_sets, plot_every_nth_point)

#make the graphs fit better 
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=.2)

plt.show()

#save the plot
plt.savefig(output_folder + 'compare_K_perm_rates_5_min_square_no_offset.png', format='png', dpi=200)

conditions_to_plot=['single square wave permK=0', 'single square wave permK=normal', 
                    'single square wave permK=fast', 'single square wave permK=very fast'] #, 'single square wave permK=very fast']
plot_font_size=6

fig = plt.figure(num='Figure 5 (as in paper)', figsize=(4,2), dpi=200) #make a figure
plt.rcParams.update({'font.size': plot_font_size})
ax1=fig.add_subplot(131)
ax2=fig.add_subplot(132)
ax3=fig.add_subplot(133)


use_colors=['red', 'orange','green', 'blue']
labels_text=['P=0', 'P=normal', 'P=10X', 'P=100X']

for index, con in enumerate(conditions_to_plot):
    output=output_dict[con]
    time_axis=np.array(output['time_axis'])
    time_offset=2.5*60
    
    #ax1.plot(time_axis[b:e]-time_offset, output['K_flux'][b:e], label=con, lw=1, color=use_colors[index])
    ax1.plot(time_axis-time_offset, output['K_flux'], label=con, lw=1, color=use_colors[index])

    
    ax2.plot(time_axis-time_offset, output['deficit']*(4.67), 
        label=labels_text[index], lw=1, color=use_colors[index])

    ax3.plot(time_axis-time_offset, output['deficit_int']/3.0, 
        label=labels_text[index], lw=1, color=use_colors[index])

ax1.set_xlim(-1,30)
ax1.set_ylim(-20,420)
ax1.set_ylabel(r'K$^+$ flux (mol mol$^{-1}$ PSII s$^{-1}$)', size=plot_font_size, labelpad=1)
ax1.set_xlabel(r'time (s)', size=plot_font_size, labelpad=1)

ax2.set_xlim(-1,30)
ax2.set_ylim(-20,420)

ax2.set_ylabel(r'proton deficit (fraction)', size=plot_font_size, labelpad=1)
ax2.set_xlabel(r'time (s)', size=plot_font_size, labelpad=1)

ax3.set_ylabel(r'ATP deficit (cumulative)', size=plot_font_size, labelpad=1)
ax3.set_xlabel(r'time (s)', size=plot_font_size, labelpad=1)

ax3.set_xlim(-1,30)
ax3.set_ylim(-10,150)


ax2.legend(loc='upper center', bbox_to_anchor=(0.7, 0.99), fancybox=False, 
                       shadow=False, frameon=False, ncol=1,
                      fontsize=plot_font_size*.75)

ax3.set_xlim(-1,30)

props = dict(boxstyle='circle', facecolor='white', alpha=1)

sub_plot_annotation='A'
ax1.text(0.8, .2, sub_plot_annotation, transform=ax1.transAxes, fontsize=plot_font_size,
            verticalalignment='top', bbox=props)

sub_plot_annotation='B'
ax2.text(0.8, .2, sub_plot_annotation, transform=ax2.transAxes, fontsize=plot_font_size,
            verticalalignment='top', bbox=props)

sub_plot_annotation='C'
ax3.text(0.8, .2, sub_plot_annotation, transform=ax3.transAxes, fontsize=plot_font_size,
            verticalalignment='top', bbox=props)


        
plt.tight_layout()
plt.savefig(output_folder + 'ATP_deficit.png', format='png', dpi=200)

#simulate responses to a single square pulse of actinic light
#keeping all rate constants and conditions the same as standard.

on='single sin wave permK=normal' #the output name

Kx=sim_constants() #generrate arrays contining optimized time segments for the simulation
Kx.k_KEA=0
Kx.fraction_pH_effect=.25

Kx.perm_K=0.2*Kx.perm_K #multiple the perm_K value by 10

constants_dict[on]=Kx #store constants in constants_dict

output_dict[on], starting_conditions_dict[on]=sim(Kx, original_dark_equilibrated_initial_y, 
                                        light_pattern['single_sin_wave_1_hr_600_max'], 
                                        max_light_change, points_per_segment, dark_equilibration=60*60)

#plot out the "interesting features"
#plot_interesting_stuff('Test Output 7', output_dict[on])



#simulate responses to a single, one-hour sin wave of actinic light
#making perm_K 10-fold faster

on='single sin wave permK=fast' #the output name

Kx=sim_constants() #generrate arrays contining optimized time segments for the simulation
Kx.k_KEA=0
Kx.fraction_pH_effect=.25

Kx.perm_K=30*Kx.perm_K #multiple the perm_K value by 10

constants_dict[on]=Kx #store constants in constants_dict

output_dict[on], starting_conditions_dict[on]=sim(Kx, original_dark_equilibrated_initial_y, 
                                        light_pattern['single_sin_wave_1_hr_600_max'], 
                                        max_light_change, points_per_segment, dark_equilibration=60*60)

#plot out the "interesting features"
#plot_interesting_stuff('Test Output 8', output_dict[on])

#simulate responses to a single square pulse of actinic light
#keeping all rate constants and conditions the same as standard.

on='one hour square wave permK=normal' #the output name

Kx=sim_constants() #generrate arrays contining optimized time segments for the simulation
Kx.k_KEA=0
Kx.fraction_pH_effect=.25
Kx.perm_K=0.2*Kx.perm_K #multiple the perm_K value by 10

constants_dict[on]=Kx #store constants in constants_dict

output_dict[on], starting_conditions_dict[on]=sim(Kx, original_dark_equilibrated_initial_y, 
                                        light_pattern['one_hour_5_min_cycle_square_wave_max_PAR_600'], 
                                        max_light_change, points_per_segment, dark_equilibration=60*60)

#plot out the "interesting features"
#plot_interesting_stuff('Test Output 9', output_dict[on])

#simulate responses to a single square pulse of actinic light
#keeping all rate constants and conditions the same as standard.

on='one hour square wave permK=fast' #the output name

Kx=sim_constants() #generrate arrays contining optimized time segments for the simulation
Kx.perm_K=30*Kx.perm_K #multiple the perm_K value by 10
Kx.k_KEA=0
Kx.fraction_pH_effect=.25

constants_dict[on]=Kx #store constants in constants_dict

output_dict[on], starting_conditions_dict[on]=sim(Kx, original_dark_equilibrated_initial_y, 
                                        light_pattern['one_hour_5_min_cycle_square_wave_max_PAR_600'], 
                                        max_light_change, points_per_segment, dark_equilibration=60*60)

#plot out the "interesting features"
#plot_interesting_stuff('Test Output 10', output_dict[on])

#set up a list, called conditions_to_plot, that holds the sets of simulation conditions 

conditions_to_plot=['single sin wave permK=normal', 
                    'single sin wave permK=fast',
                    'one hour square wave permK=normal', 
                    'one hour square wave permK=fast']

where=[1,2,3,4] #will hold the col positions

phenomena_sets=['pmf_params_offset', 'plot_cum_LEF_singetO2']    
        

fig = plt.figure(num='Figure 4 in paper', figsize=(5,4), dpi=200) #make a figure
plot_every_nth_point=1
plot_block(output_dict, fig, conditions_to_plot, where, phenomena_sets, plot_every_nth_point)


plt.tight_layout()
plt.show()
#plt.title('Figure X')

#plt.title('Figure 4 in paper')
plt.savefig(output_folder + 'Figure 4 in paper sin and square one rows.png', format='png', dpi=200)
plt.show()

#set up a list, called conditions_to_plot, that holds the sets of simulation conditions 

conditions_to_plot=['single sin wave permK=normal', 
                    'single sin wave permK=fast',
                    'one hour square wave permK=normal', 
                    'one hour square wave permK=fast']
where=[1,2,3,4] #will hold the col positions

phenomena_sets=['pmf_params_offset', 
                'b6f_and_balance', 
                'K_and_parsing', 
                'plot_QAm_and_singletO2', 
                'plot_cum_LEF_singetO2']    
        

fig = plt.figure(num='Figure S3 (extended version of Figure 4)', figsize=(5,7), dpi=200) #make a figure
plot_every_nth_point=1
plot_block(output_dict, fig, conditions_to_plot, where, phenomena_sets, plot_every_nth_point)


plt.tight_layout()
plt.show()


plt.savefig(output_folder + 'sin and square all rows.png', format='png', dpi=200)

