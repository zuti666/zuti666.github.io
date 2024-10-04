---
layout: post
title:  Fix GitHub MathJax Display Problem
categories: [Blog Website] 
description: Fix GitHub MathJax Display Problem
keywords: [Blog, MathJax] 

---


# Fix GitHub MathJax Display Problem



## 解决 GitHub markdown公式显示问题 



### 参考链接


[让MathJax更好地兼容谷歌翻译和延时加载 - 科学空间Scientific Spaces (kexue.fm)](https://kexue.fm/archives/10320)

解决方法：

1 块状公式 $$ $$ 前后一定加一行空行

2 在 _includes/header.html 中添加下述代码

```html
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']], displayMath: [['$$','$$'], ['\\[','\\]']]   },
        TeX: {equationNumbers: {autoNumber: ["AMS"], useLabelIds: true}, extensions: ["AMSmath.js", "AMSsymbols.js", "extpfeil.js"]},
        "HTML-CSS": {linebreaks: {automatic: true, width: "95% container"}, noReflows: false, availableFonts: ["tex"], styles: {".MathJax_Display": {margin: "1em 0em 0.7em;", display: "inline-block!important;"}}},
        "CommonHTML": {linebreaks: {automatic: true, width: "95% container"}, noReflows: false, availableFonts: ["tex"], styles: {".MJXc-display": {margin: "1em 0em 0.7em;", display: "inline-block!important;"}}},
        "SVG": {linebreaks: {automatic: true, width: "95% container"}, styles: {".MathJax_SVG_Display": {margin: "1em 0em 0.7em;", display: "inline-block!important;"}}},
        "PreviewHTML": {linebreaks: {automatic: true, width: "95% container"}}
    });
    MathJax.Hub.Queue(function() {
        document.querySelectorAll('.MathJax').forEach(element => element.classList.add('notranslate'));
        document.querySelectorAll('a.title-link, p.summary').forEach(element => element.classList.remove('notranslate'));
    });
</script>
<script src="/static/MathJax-2.7.9/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
```



##  解决第一页 文章摘要显示过多的问题

解决方法 

在index.html中 第106行

添加截断文本显示的代码 , | truncate: 100 

```html
<p class="repo-list-description">
  {{ post.excerpt | strip_html | strip | truncate: 100 }}
</p>
```

## 代码高亮

首先修改Jekyll项目的**_config.yml** 文件，修改其中的 highlighter 为：rouge

```html
# markdown 设置
markdown: kramdown
kramdown:
  math_engine: mathjax
  syntax_highlighter: rouge
```



然后在_includes/header.html中添加以下代码 

```html
 <!-- 代码高亮 -->
    <link rel="stylesheet" href="{{ assets_base_url }}/assets/css/posts/github.css">
    <script src="{{ assets_base_url }}/assets/js/highlight.min.js"></script>
```

同时在对应的路径文件夹中添加对应的css 文件和 js 文件

js 下载地址 ：  [Download a Custom Build - highlight.js (highlightjs.org)](https://highlightjs.org/download)

