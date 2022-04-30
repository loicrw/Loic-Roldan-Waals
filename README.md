# Random-problems
This is a collection of Random python problems that I decided to solve

Sometimes there are problems installing the 'pygraphviz' library. If so, follow these instructions:
```
#install graphviz first
brew install graphviz

#check your graphviz path   
brew info graphviz

#change to your dir
export GRAPHVIZ_DIR="/usr/local/Cellar/graphviz/<VERSION>" #3.0.0 in my case

#finally run this 
pip install pygraphviz --global-option=build_ext --global-option="-I$GRAPHVIZ_DIR/include" --global-option="-L$GRAPHVIZ_DIR/lib"``` 