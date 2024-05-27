## In this project, we've implemented EDA on scraped data from [CIAN](https://www.cian.ru/), which includes properties and prices of apartments located in Moscow. The price range that we considered to analyse was in range 0 - 40 millions of rubles.

#  Suddenly we recognised that we need to provide user-friendly UI for any observers. All of our work you can view via [link](https://cianpredict-djezjgrq2se8qwq2yupeqe.streamlit.app/).

## Now, after reviewing our work, let's figure out the structure of the project.

## There are 5 logical parts inside our repo👇
- data (on different stages of our project we were addicted to use different type of data preprocess)
- notebooks (consists of notebooks we were using)
- pages (all pages for streamlit_app indeed)
- src (in this directory we've implemented all of our work via python classes to allow the user to repeat/to check all stages of the project) 
- weights (fitted catboost regressor model inside)

As you could see we used poetry. That was necessary to install all dependencies for making streamlit report
️ 
