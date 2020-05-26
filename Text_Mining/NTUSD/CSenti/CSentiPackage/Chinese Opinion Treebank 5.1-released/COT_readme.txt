
Chinese Opinion Treebank Version 1.0 Readme Last Update December 9, 2016

1. Directory:
Chinese Opinion Treebank 5.1-released
	inter-word relations(trios)/
		[docID]/
			[sentenceID]/
				[docID]_[sentenceID].tree: the parsing tree of the sentence
				[docID]_[sentenceID].tri: the labeled trios on this sentence
					
	opinion/
		sentence.csv: the opinion labels of sentences

The docID and sentenceID can be used to aligh with the Chinese Treebank 5.1.
Note that the sentenceID is continuous, that is, it is not reset to 1 for each document.

2. File Format:		

2.1 The format of each line in the *.tree file is 

[parent_node]:[child_node1],[child_node2],...[child_nodeN], where for this parent_node, there are N child nodes.
The root node ID is 0.

EX:

In the tree

0:1,
1:2,6,

Word 0 is the root, Word 1 is its only child.
Word 2 and 6 are the children of the Word 1.



2.2 The format of each line in the *.tri file is 


[trioID],[trioParent],[trioLeft],[trioRight],[trioType]
trioID starts from 0.

EX:

In the tio
0,8,15,17,2
The trio is composed of three words: Word 8, 15, 17, and Word 8 is the parent, 
Word 15 is the left child, and Word 17 is the right child.
This trio is of type 2, where the trio ID represents the following types:

1: Parallel
2: Substantive-Modifier
3: Subjective-Predicate
4: Verb-Object
5: Verb-Complement


2.3 The format of each row in the sentence.csv file is 

chtb_[docID].raw,[sentenceID],[Opinion],[Polarity],[Type]
The column [Opinion] is of two possible values, Y and N, 
indicate an opinion sentence and a non-opinion sentence respectively. 

The column [Polarity] is of three possible values, POS for positive, NEU for neutral, and NEG for negative.

The column [Type] is of three possible values:
ACTION: this sentence is expressing opinions in the form of an action
STATE: this sentence is expressing opinions by telling the current state
PEOPLE: this sentence is expressing opinions by the judgement of people.

EX.

chtb_001.raw, 3, Y, POS, ACTION denotes that the third sentence in the whole dataset, which is in chtb_001.raw, 
is an opinion sentence of the positive polarity. It expresses an opinion by describing an action.

 







