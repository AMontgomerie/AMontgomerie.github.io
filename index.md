<h1>Latest Post</h1>
{% for post in site.posts limit:1 %}
{{ post.content }}
{% endfor %}

<h1>Recent Posts</h1>
{% for post in site.posts offset:1 %}
 <li>
   <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
   {{ post.excerpt }}
 </li>
{% endfor %}

