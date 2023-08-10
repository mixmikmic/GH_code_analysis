# import required libraries
import re
import requests

def extract_blog_content(content):
    """This function extracts blog post content using regex

    Args:
        content (request.content): String content returned from requests.get

    Returns:
        str: string content as per regex match

    """
    content_pattern = re.compile(r'<div class="cms-richtext">(.*?)</div>')
    result = re.findall(content_pattern, content)
    return result[0] if result else "None"

base_url = "http://www.apress.com/in/blog/all-blog-posts"
blog_suffix = "/wannacry-how-to-prepare/12302194"

response = requests.get(base_url+blog_suffix)

if response.status_code == 200:
        content = response.text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
        content = content.replace("\n", '')
        blog_post_content = extract_blog_content(content)

blog_post_content[0:500]

