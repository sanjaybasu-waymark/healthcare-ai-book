# Gemfile for Modern Technical Book on GitHub Pages
source "https://rubygems.org"

# GitHub Pages gem for compatibility
gem "github-pages", group: :jekyll_plugins

# Modern theme for technical documentation
gem "just-the-docs"

# Essential Jekyll plugins for technical books
group :jekyll_plugins do
  gem "jekyll-feed"
  gem "jekyll-sitemap"
  gem "jekyll-seo-tag"
  gem "jekyll-github-metadata"
  gem "jekyll-relative-links"
  gem "jekyll-optional-front-matter"
  gem "jekyll-readme-index"
  gem "jekyll-default-layout"
  gem "jekyll-titles-from-headings"
  gem "jekyll-jupyter-notebook"

  gem "jekyll-toc"
end

# Additional gems for enhanced functionality
gem "kramdown-parser-gfm"
gem "rouge"
gem "webrick", "~> 1.7"

# Development and testing
group :development, :test do
  gem "html-proofer"
  gem "jekyll-livereload"
end

# Platform-specific gems
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

# Performance-booster for watching directories on Windows
gem "wdm", "~> 0.1.1", :platforms => [:mingw, :x64_mingw, :mswin]

# Lock `http_parser.rb` gem to `v0.6.x` on JRuby builds
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby]
