Snowball: Extracting Relations from Large Plain-Text Collections
================================================================

This is my own implementation of the the Snowball system to bootstrap relationship instances. You can find more details here: 

- Eugene Agichtein and Luis Gravano, [Snowball: Extracting Relations from Large Plain-Text Collections](http://www.mathcs.emory.edu/~eugene/papers/dl00.pdf). In Proceedings of the fifth ACM conference on Digital libraries. ACM, 200.

- H Yu, E Agichtein, [Extracting synonymous gene and protein terms from biological literature](http://bioinformatics.oxfordjournals.org/content/19/suppl_1/i340.full.pdf). In Bioinformatics, 19(suppl 1), 2003 - Oxford University Press

A sample file containing sentences where the named-entities are already tagged can [downloaded](https://drive.google.com/open?id=0B0CbnDgKi0PyM1FEQXJRTlZtSTg), which has 1 million sentences taken from the New York Times articles part of the English Gigaword Collection.

**NOTE**: look at the desription of [BREDS](https://github.com/davidsbatista/BREDS) to understand how to give a tagged document collection and seeds to setup the bootstrapping of relationship instances with Snowball, both systems have a similar setup.
