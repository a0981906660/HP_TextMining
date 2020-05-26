******************************
CopeOpi Readme Last Update Dec. 7, 2016
By Lun-Wei Ku
******************************

CopeOpi (Traditional Version and Simplified Version)

0. Preprocessing 

0.1 Environment


0.2 Text preprocessing
You text must be encoded by unicode.
Your text must be segmentated *AND* part-of-speeech tagged by the

A. For traditional Chinese: CKIP POS tagger 
http://ckipsvr.iis.sinica.edu.tw/

*OR*

B. For simplified Chinese: Stanford CoreNLP POS tagger 
http://stanfordnlp.github.io/CoreNLP/
Then for CopeOpi to process texts in simplied Chinese, the file "penn2ckip.class" must be run first to map Stanford format to CKIP format.
The POS mapping file "map.txt" lists how to map the Stanford tagging set and the CKIP tagging set. 
You can update this mapping by yourself when necessary.
If there is any tag which is not found in the map.txt, the original tag will be kept and CopeOpi will not consider words of this tag when analyzing 
sentiment.



1. Directories

Root: CopeOpi_released/  
	dic_trad/ (traditional Chinese)
		pos_unigram.txt: sentiment score and confident score for positive Chinese characters
		neg_unigram.txt: sentiment score and confident score for negative Chinese characters
		positive_new.txt: positive Chinese sentiment words
		negative_new.txt: negative Chinese sentiment words
		negation.txt: negation words 	

	dic_simp/ (simplified Chinese)
		pos_unigram.txt: sentiment score and confident score for positive Chinese characters
		neg_unigram.txt: sentiment score and confident score for negative Chinese characters
		positive_new.txt: positive Chinese sentiment words
		negative_new.txt: negative Chinese sentiment words
		negation.txt: negation words 	

	opinion/
		OpinionCore_Enhanced.jar: a jar file including all opinion functions
	CopeOpi_trad.java: an example main function using opinion fuctions (traditional Chinese)
	CopeOpi_trad.class: the compiled file of CopeOpi.java (traditional Chinese)

	CopeOpi_simp.java: an example main function using opinion fuctions (simplified Chinese)
	CopeOpi_simp.class: the compiled file of CopeOpi.java (simplified Chinese)

	test_trad.txt: input example file (traditional Chinese)
	test_simp.txt: input example file (simplified Chinese)

	readme.txt: this file
	map.txt: the POS mapping file from Stanford format to CKIP format
	run_penn2ckip.sh
	run_trad.sh: input example script (traditional Chinese)
	run_simp.sh: input example script (simplified Chinese)
	 																						 


2. Input

The pure text content (text file) to be analyzed as test_trad.txt or test_simp.txt

3. Output

Output to the standard output. If with the execusion argument -n, the overall opinion score and the polarity are printed out; 
if with the execusion argument -d, the content of each sentence and its corresponding score will be printed out.

4. Excusion

java CopeOpi [Input Filename List] [Output Arguments]

EX: java -cp "opinion/*.jar:." CopeOpi_trad file_trad.lst -n/d (traditional Chinese)
    java -cp "opinion/*.jar:." CopeOpi_simp file_simp.lst -n/d (simplified Chinese)
    
5. Discriptions of Dictionaries

5.1 Files for character opinion scores:
Format:
	Character	Sentiment-Frequency	Overall-Frequency	Score		Confidence	
EX:        愛		29			30			0.986123	0.90

The current module use the score only,
confidence score is only for reference.

5.2 Sentiment word lists: positive_new.txt and negative_new.txt

Format
(Part-Of-Speech);Word
EX:
(無);一夕成名

If the part of speech of the sentiment word is set to "無", this word is always treated as the sentiment word
in the program no matter what its part of speech is.

5.3 Negation file: negation.txt

All the words included in this file will be treated as negation words.




