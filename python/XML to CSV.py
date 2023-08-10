sc

#funtion for CSV conversion
def line22csv(line,tags_list):
    offset=0
    result=""
    for i in tags_list:
        val=""
        patt=i+"="
        ind=line.find(patt,offset)
        if(ind==-1):
            result+=','
            continue
        ind+=(len(i)+2)
        val+='\"'
        while(line[ind]!='\"'):
            val+=line[ind]
            ind+=1
        val+='\"'
        result+=val+','
        offset=ind
    return result[:-1]

fileName = 'Users.xml'

raw = (sc.textFile(fileName, 4))

#Removing top 2 lines form XML file,they didn't contain useful data
headers = raw.take(2)
UsersRDD = raw.filter(lambda x: x != headers)

#FieldNames for Users
tags_list=['Id','Reputation','CreationDate','DisplayName','LastAccessDate',
           'WebsiteUrl','Location','AboutMe','Views','UpVotes','DownVotes',
           'EmailHash','AccountId','Age']

Users_csvRDD=UsersRDD.map(lambda x:line22csv(x,tags_list))

#Folder path to save processed files
targetFile = './users_csv'

Users_csvRDD.saveAsTextFile(targetFile)

#FieldNames for Posts
tags_list=['Id','PostTypeId','AcceptedAnswerId','ParentId','CreationDate',
           'DeletionDte','Score','ViewCount','Body','OwnerUserId','OwnerDisplayName',
           'LastEditorUserId','LastEditorDisplayName','LastEditDate','LastActivityDate',
           'Title','Tags','AnswerCount','CommentCount','FavoriteCount','ClosedDate',
           'CommunityOwnedDate']

fileName = 'Posts.xml'

raw = (sc.textFile(fileName, 4))

#Removing top 2 lines form XML file,they didn't contain useful data
headers = raw.take(2)
PostsRDD = raw.filter(lambda x: x != headers)

Posts_csvRDD=PostsRDD.map(lambda x:line22csv(x,tags_list))

targetFile = './posts_csv'

Posts_csvRDD.saveAsTextFile(targetFile)



