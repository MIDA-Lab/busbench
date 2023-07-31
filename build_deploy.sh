docker run --rm -v $PWD/src:/srv/jekyll jekyll/jekyll jekyll b
scp -r src/_site mida:/data/busbench
