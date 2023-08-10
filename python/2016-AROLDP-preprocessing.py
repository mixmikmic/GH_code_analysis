def read_file(filepath):
    try:
        filepath.index("/");
        array = filepath.split('/');
        filename = array[len(array) - 1]
    except:
        filename = filepath
    print(filename)
    ## filename = UMAFall_Subject_01_ADL_Bending_1_2016-06-13_20-25-34.csv
    array = filename.split('_')
    subjectID = array[1] + " " + array[2]
    label = array[4]
    experiment = array[4] + "_" + array[5]
    ### read csv
    train_dataRaw = pd.read_csv(filepath)
    ### remove first 31 lines
    train_data = train_dataRaw.iloc[31:]
    ### user characteristics
    user = SUBJECTS[SUBJECTS['id'].isin([subjectID])];
    age = user['age']
    gender = user['gender']
    height = user['height']
    weight = user['weight']

    return {"filecontent": train_data, 
             'experiment': experiment, 
             'label': label, 
             'subjectID': subjectID,
             'filename': filename, 
             'gender': gender.values[0], 
             'age': age.values[0], 
             'height': height.values[0], 
             'weight': weight.values[0]};
   

###output = read_file("../../../datasets/ANSAMO-2016/UMA_ADL_FALL_Dataset/UMAFall_Subject_01_ADL_Bending_1_2016-06-13_20-25-34.csv");

