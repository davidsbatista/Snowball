Snowball: Extracting Relations from Large Plain-Text Collections
================================================================

This is my own implementation of the the Snowball system to bootstrap relationship instances. You can find more details about the original system here: 

- Eugene Agichtein and Luis Gravano, [Snowball: Extracting Relations from Large Plain-Text Collections](http://www.mathcs.emory.edu/~eugene/papers/dl00.pdf). In Proceedings of the fifth ACM conference on Digital libraries. ACM, 200.

- H Yu, E Agichtein, [Extracting synonymous gene and protein terms from biological literature](http://bioinformatics.oxfordjournals.org/content/19/suppl_1/i340.full.pdf). In Bioinformatics, 19(suppl 1), 2003 - Oxford University Press


For more details about this particular implementation please refer to:

- David S Batista, Bruno Martins, and Mário J Silva. , [Semi-Supervised Bootstrapping of Relationship Extractors with Distributional Semantics](http://davidsbatista.net/assets/documents/publications/breds-emnlp_15.pdf). In Empirical Methods in Natural Language Processing. ACL, 2015. (Honorable Mention for Best Short Paper)

- David S Batista, Ph.D. Thesis, [Large-Scale Semantic Relationship Extraction for Information Discovery (Chapter 5)](http://davidsbatista.net/assets/documents/publications/dsbatista-phd-thesis-2016.pdf), Instituto Superior Técnico, University of Lisbon, 2016



A sample file containing sentences where the named-entities are already tagged can be [downloaded](https://drive.google.com/open?id=0B0CbnDgKi0PyM1FEQXJRTlZtSTg), which has 1 million sentences taken from the New York Times articles part of the English Gigaword Collection.

**NOTE**: look at the desription of [BREDS](https://github.com/davidsbatista/BREDS) to understand how to give a tagged document collection and seeds to setup the bootstrapping of relationship instances with Snowball, both systems have a similar setup.
