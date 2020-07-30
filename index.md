<ul>
   {% for item in site.data.navigation %}
      <li><a href="{{ item.url }}">{{ item.title }}</a></li>
   {% endfor %}
</ul>

<h1>Latest Post</h1>
{% for post in site.posts limit:1 %}
<h1>{{ post.title }}</h1>
{{ post.content }}
{% endfor %}

<h1>Recent Posts</h1>
{% for post in site.posts offset:1 limit:2 %}
<h1>{{ post.title }}</h1>
{{ post.content }}
{% endfor %}
