<ul>
   {% for item in site.data.navigation %}
      <a href="{{ item.url }}">{{ item.title }}</a>
   {% endfor %}
</ul>

<h1>Latest Post</h1>
{% for post in site.posts limit:1 %}
<h1>{{ post.title }}</h1>
{{ post.content }}
{% endfor %}

<h1>Recent Posts</h1>
{% for post in site.posts offset:1 %}
<h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
{{ post.excerpt }}
{% endfor %}

<ul>
  {% for post in site.posts %}
    <li>
      <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
      {{ post.excerpt }}
    </li>
  {% endfor %}
</ul>
