package cs224n.assignment;

import java.util.List;
import cs224n.ling.Tree;

/**
 * Parser interface
 *
 * @author Dan Klein
 */

/**
 * Parsers are required to map sentences to trees.  How a parser is
 * constructed and trained is not specified.
 */
public interface Parser {
    public void train(List<Tree<String>> trainTrees);
    public Tree<String> getBestParse(List<String> sentence);
}
