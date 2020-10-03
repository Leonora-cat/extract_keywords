# Extract_Keywords

This project is a complete program containing front-end and back-end, where users could upload a file in a web page, and then keywords in the file would be extracted and ready to download.



## Method

The back-end of this project uses 4 ways to extract keywords: TF-IDF, word2vec_kmeans, word2vec_huffman, and chi squared test. 

However, due to performance reason, only TF-IDF,  word2vec_huffman, and chi squared test are provided on the web page. Moreover, user can also download the intersection keywords of these 3 methods.

The front-end is developed just for better present this project. Thus, the web page is very simple and not beautiful.



## Result

With the data from actual scene, word2vec_huffman, and chi squared test turned out to be 2 satisfactory methods considering the quality of extracted keywords. 

However, word2vec_huffman took much longer time than other methods in spite of its high-quality result. To reduce time consumption, multiprocessing is employed and the running time has dropped by approximately 50%. 

Generally chi squared test showed best performance among 4 methods, taking into account its small time consumption and relatively good result. However, chi squared test did not consider the importance of word frequency, which means that what mattered was whether a word appeared instead of how many times a word appeared. Hence, this method could be further improved.

In addition, jieba tokenization is quite time-consuming if the data file is large. Similarly, multiprocessing is used and the time to tokenize has dropped by approximately 30%.
