DATA="data/tweets"

get_ipython().system("head -1 $DATA | jq '.'")

get_ipython().system("head -1 $DATA| jq '.[]'")

get_ipython().system("head -3 $DATA | jq '[.created_at, .text]'")

get_ipython().system("head -1 $DATA | jq '[.user]'")

get_ipython().system("head -2 $DATA | jq '[.user.screen_name, .user.name, .user.followers_count, .user.id_str]'")

get_ipython().system('cat $DATA | jq \'[([.entities.hashtags[].text] | join(","))]\'')

get_ipython().system("head -8 $DATA | jq -r '[.id_str, .created_at, .text] | @csv'")

get_ipython().system("cat $DATA | jq -r '[.id_str, .created_at, .text] | @csv' > tweets.csv")

get_ipython().system('head tweets.csv')

get_ipython().system('cat $DATA | jq -r \'[.id_str, .created_at, (.text | gsub("\\n";" "))] | @csv\' > tweets-oneline.csv')

get_ipython().system('head tweets-oneline.csv')

get_ipython().system("cat $DATA | jq -c '{{id: .id_str, user_id: .user.id_str, screen_name: .user.screen_name, created_at: .created_at, text: .text, user_mentions: [.entities.user_mentions[]?.screen_name], hashtags: [.entities.hashtags[]?.text], urls: [.entities.urls[]?.expanded_url]}}' > newtweets.json")

get_ipython().system('head newtweets.json')



