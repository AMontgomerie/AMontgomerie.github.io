<h1>Latest Post</h1>
{% for post in site.posts limit:1 %}
{{ post.content }}
{% endfor %}

