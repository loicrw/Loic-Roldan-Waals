This is a collection of Random python problems that I decided to solve

# Projects
Here are the links to each of the projects that I've worked on:
- [The impostor problem](https://loicrw.github.io/Random-problems/random_problems/the_impostor_problem/)
- [The best hangman word]()
- [Predicting personal movie ratings with IMDb data]()
- [The longest 7-cell display word]()
- [Python teaching material]()
- [Special digital clock times]()
- [Toilet paper numbers]()

Sometimes there are problems installing the 'pygraphviz' library. If so, follow these instructions:

```python
#install graphviz first
brew install graphviz

#check your graphviz path   
brew info graphviz

#change to your dir
export GRAPHVIZ_DIR="/usr/local/Cellar/graphviz/<VERSION>" #3.0.0 in my case

#finally run this 
pip install pygraphviz --global-option=build_ext --global-option="-I$GRAPHVIZ_DIR/include" --global-option="-L$GRAPHVIZ_DIR/lib"
``` 

You can find the other topics here [example](https://github.com/loicrw/Random-problems/blob/master/Random%20Problems/best%20hangman%20word/google-10000-english-master/README.md)
