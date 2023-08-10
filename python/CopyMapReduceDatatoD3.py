#Copy the src_file (word count) to dst_file in js (d3 readable file)
def create_js_file(src_file, dst_file, objname):
    newLine = "var " + objname + " = ["
    dst_file.write(newLine + "\n")
    for line in src_file.readlines():
        line = line.strip()
        word = line.split(',')
        newLine = "{text:'" + word[0] + "', size:" + word[1] + "},"
        dst_file.write(newLine + "\n")
    dst_file.write("];")
    
#Copy the src_file (co-occurrent word output) to dst_file in js (d3 readable file)
def create_co_js_file(src_file, dst_file, objname):
    newLine = "var " + objname + " = ["
    dst_file.write(newLine + "\n")
    for line in src_file.readlines():
        line = line.strip()
        word = line.split(',')
        newLine = "{text:'" + word[0] + "-" + word[1] + "', size:" + word[2] + "},"
        dst_file.write(newLine + "\n")
    dst_file.write("];")

print ("Copying word count of twitter from /hadoop/twitter/twt_wrdcnt_one_week/part-00000")
srcfile = open("./hadoop/twitter/twt_wrdcnt_one_week/part-00000", "r")
newFile = open("./d3_wordcloud/data/part-0000-twt-wc.js", "w")
create_js_file(srcfile, newFile, "tweet_words")
srcfile.close()
newFile.close()
print("Copied word count of twitter data to /d3_wordcloud/data/part-0000-twt-wc.js in js format")

print ("----")

print ("Copying word count of nytime article from /hadoop/ny_article/nyt_wrdcnt_one_week/part-00000")
srcfile = open("./hadoop/ny_article/nyt_wrdcnt_one_week/part-00000", "r")
newFile = open("./d3_wordcloud/data/part-0000-nyt-wc.js", "w")
create_js_file(srcfile, newFile, "nyt_words")
srcfile.close()
newFile.close()
print("Copied word count of nytimes data to /d3_wordcloud/data/part-0000-nyt-wc.js in js format")

print ("----")

print ("Copying word count of twitter from /hadoop/twitter/twt_cooccur_one_week/part-00000")
srcfile = open("./hadoop/twitter/twt_cooccur_one_week/part-00000", "r")
newFile = open("./d3_wordcloud/data/part-0000-twt-co.js", "w")
create_co_js_file(srcfile, newFile, "tweet_co_words")
srcfile.close()
newFile.close()
print("Copied word count of twitter data to /d3_wordcloud/data/part-0000-twt-co.js in js format")

print ("----")

print ("Copying word count of nytime article from /hadoop/ny_article/nyt_cooccur_one_week/part-00000")
srcfile = open("./hadoop/ny_article/nyt_cooccur_one_week/part-00000", "r")
newFile = open("./d3_wordcloud/data/part-0000-nyt-co.js", "w")
create_co_js_file(srcfile, newFile, "nyt_co_words")
srcfile.close()
newFile.close()
print("Copied word count of nytimes data to /d3_wordcloud/data/part-0000-nyt-co.js in js format")


