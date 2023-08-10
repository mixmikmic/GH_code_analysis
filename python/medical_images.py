import numpy as np
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import dicom
import subprocess

get_ipython().run_line_magic('pylab', 'inline')

def partitioned_var(list_stds,list_elts):
    stds=[]
    for i in xrange(0,len(list_stds)):
        stds.append(list_elts[i]*list_stds[i])
    return np.sum(stds)/np.sum(list_elts)

def partitioned_mean(list_means,list_elts):
    means=[]
    for i in xrange(0,len(list_stds)):
        means.append(list_elts[i]*list_means[i])
    return np.sum(means)/np.sum(list_elts)

def combine_organs_GT(list_name_organs):
    #list of sitk images
    list_images=[]
    for fname in list_name_organs:
        list_images.append(sitk.ReadImage(fname))
    
    size=list_images[0].GetSize()
    output_image = sitk.Image(size,sitk.sitkUInt8)
    
    list_np=[]
    for img in list_images:
        list_np.append(sitk.GetArrayFromImage(img))
    cnt=1
    output_np=np.zeros_like(list_np[0])
    for img in list_np:
        #print img.shape
        output_np[np.where(img==1)]=cnt
        cnt+=1
    
    print np.unique(output_np)
    output_image=sitk.GetImageFromArray(output_np)
    output_image.CopyInformation(list_images[0])
    return output_image
        
        
        
                       
    

plastimatch_path='./Users/rogertrullo/Documents/Plastimatch/plastimatchbin/plastimatch'
dirpatients='./DOI/'
output_dir='/Users/rogertrullo/Documents/Lung_CT_challenge'
path, dirs, files = os.walk(dirpatients).next()#every folder is a patient
n_total_voxels=0
list_means=[]
list_stds=[]
list_elts=[]
list_imgs=[]

list_name_organs=['SpinalCord.mha', 'Esophagus.mha', 'Lung_L.mha', 'Heart.mha', 'Lung_R.mha']

if not os.path.exists(output_dir):
        os.mkdir(output_dir)


for d in dirs:
    print d
    path_, dirs_, files_ = os.walk(os.path.join(path,d)).next()#every folder is a patient
    _,dirs_final,_=os.walk(os.path.join(path,d,dirs_[0])).next()#every folder is a patient
    path_1=os.path.join(path,d,dirs_[0],dirs_final[1])
    path_2=os.path.join(path,d,dirs_[0],dirs_final[0])
    nfiles1=os.listdir(path_1)
    nfiles2=os.listdir(path_2)
    if len(nfiles1)>1:
        path_ct=path_1
        path_gt=path_2
    else:
        path_ct=path_2
        path_gt=path_1
        
    reader = sitk.ImageSeriesReader()
    #print path_ct
    reader.GetGDCMSeriesFileNames(path_ct)
    dicom_names = reader.GetGDCMSeriesFileNames( path_ct )
    #print dicom_names
    reader.SetFileNames(dicom_names)
    image = reader.Execute() 
    image_np=sitk.GetArrayFromImage(image)
    size = image.GetSize()
    print( "Image size:", size[0], size[1], size[2])
    list_means.append(np.mean(image_np))
    list_stds.append(np.std(image_np))
    list_elts.append(np.prod(size))
    #list_imgs.extend(image_np.ravel())
    #use plastimatch to convert RT-Dicom to image
    #path_rt=os.path.join(path_gt,os.listdir(path_gt)[0])   
    
    #cmd='plastimatch convert --input {0} --output-type=uchar --referenced-ct {1} --output-prefix {2}'.format(path_rt,path_ct,'./')
    #print subprocess.check_output(cmd, shell=True)
    #output_itk=combine_organs_GT(list_name_organs)
    
    #out_folder=os.path.join(output_dir,d)
    #if not os.path.exists(out_folder):
    #    os.mkdir(out_folder)
    #sitk.WriteImage(image,os.path.join(out_folder,'{}.nii.gz'.format(d)))
    #sitk.WriteImage(output_itk,os.path.join(out_folder,'GT.nii.gz'))
    print d,'finished'

print 'everything done'
print 'partitioned mean',partitioned_mean(list_means,list_elts)
print 'partitioned std',partitioned_var(list_stds,list_elts)

#print 'real mean',np.mean(list_imgs),'partitioned mean',partitioned_mean(list_means,list_elts)
#print 'real std',np.std(list_imgs),'partitioned std',partitioned_var(list_stds,list_elts)
    

ctnp=sitk.GetArrayFromImage(image)
print ctnp.dtype,ctnp.shape,np.min(ctnp),np.max(ctnp)
slice_ct=ctnp[80,:,:]
plt.imshow(slice_ct,cmap='gray')

plt.hist(ctnp.ravel())

gt_itk=sitk.ReadImage('../GT_13.nii.gz')
gtnp=sitk.GetArrayFromImage(gt_itk)
print np.unique(gtnp)

print path_gt
print path_ct
print os.listdir(path_gt)[0]
ds=dicom.read_file(os.path.join(path_gt,os.listdir(path_gt)[0]))

rs= ds.StructureSetROISequence
for i in rs:
    print i.ROIName

contour_esophagus=ds.ROIContourSequence[0]
for k in contour_esophagus.keys():
    print contour_esophagus[k]



