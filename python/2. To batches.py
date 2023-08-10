import os
batch_size = 100
input_dir = '/home/irteam/users/data/CNN_DailyMail/cnn/1.stories_tokenized/'
output_dir = '/home/irteam/users/data/CNN_DailyMail/cnn/2.stories_tokenized_100/'

file_list = os.listdir(path=input_dir)
files_read = 0
total_files = len(file_list)
batch_count = 0

while files_read<total_files:
    batch_list = file_list[files_read:min(files_read+batch_size,total_files)]
    batch_count+=1
    files_read+=len(batch_list)
    batch_texts = []
    for file_name in batch_list:
        # read text
        with open(os.path.join(input_dir,file_name)) as f:
            text = f.read()
        # preprocess text
        text = text.replace("\n\n",' ')
        text = text.replace("  ",' ')
        text = text.replace("  ",' ')
        text = text.lower()
        # split to body and summaries
        text = text.split('@highlight')
        body = text[0].strip()
        summaries = text[1:]
        sum_out = []
        for summary in summaries:
            summary = summary.strip()
            summary = summary + ' .'
            sum_out.append(summary)
        summaries = ' '.join(sum_out)
        batch_texts.append(body+":==:"+summaries)
    batch_texts = '\n\n'.join(batch_texts)
    with open(os.path.join(output_dir,'batch_%d.txt'% batch_count),'w') as f:
        f.write(batch_texts)
    print("Saving batch %d" % batch_count)



