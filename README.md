<!DOCTYPE html>
<html>
<body>

<h2>Extracting topics in Paralign users' texts</h2>
<q> Shiladitya Chakraborty and Erin Craig </q>
<p>
Every day, Paralign community members connect with one another by sharing what&rsquo;s on their mind. So we wondered: as a whole, what does the Paralign community talk about?

To answer this question, we used a tool called <q>topic modeling</q>. The goal of topic modeling is to convert text into a mixture of its topics or themes. For example, the phrase <q>The sun is shining and I&rsquo;m in love!</q> might be represented as 60% about <q>love</q> and 40% about <q>day</q>.

So, how do we break text into its topics? We start simply: we count the words that appear in each text. For example, the texts:
<p>

<q>The sun is shining and I&rsquo;m in love!</q>
</p>
<p>
<q>The sun is covered by clouds.</q>
</p>
<p>
would become:
</p>
<img src="Blog/textGrid1.png" alt="text1" style="width:400px;display:block;">

<p>
Then, for each word and text combination, we compute how many times that word appears in that text divided by the number of times the word appears in all the texts. This gives us a measure of how frequently each word appears in each text, relative to how commonly used the word is across all the texts. So, we update our table: we divide the counts for the word <q>sun</q> by 2, which is the number of times the word <q>sun</q> appears total in all texts. (The rest of the table remains the same, as all other words appear just once.)
</p>

<img src="Blog/textGrid2.png" alt="text2" style="width:400px;">

<p>
The table we have just made is called the term frequency-inverse document frequency matrix (TF-IDF).This matrix has the texts and word frequencies, but what we really want is a matrix of texts and topics. So, we think of our TF-IDF matrix as holding <q>hidden</q> information about topics. To get at the <q>hidden</q> topic information, we break the TF-IDF matrix into a product of two matrices: one matrix will hold texts and topics, and the other matrix will hold topics and words.
</p>
<img src="Blog/matrixDecomposition.png" alt="NMF" style="width:400px;">

<p>
We break our TF-IDF matrix into a product of two matrices using a technique called Non-negative Matrix Factorization, or NMF.
</p>

<p>
Here is an example of our topic modeling in action, using examples from the Paralign community and visualized with <a href="https://pyldavis.readthedocs.io/en/latest/readme.html"> pyLDAvis.</a>
</p>

<iframe src="Blog/Top7uni.html" width="2000" height="800"></iframe> 
<p>
The top topics are:  'life', 'good', 'love', 'feel', 'day'
</p>

<h2>Mood-dependent topics</h2>
<p>
Moods and thoughts are related to one another; for example, a challenging day likely
causes a bad mood. So, we repeated our topic modeling looking at thoughts grouped
by mood. For example, we looked at the topics generated by people who reported
their mood as <q>happy</q>, <q>nervous</q>, and so on. Our results were not surprising: the
most prevalent words and topics within the <q>sad</q> mood include <q>tired</q>, <q>miss</q>, and <q>lonely</q>,
while for <q>happy</q> moods, we find words like <q>happy</q>, and <q>good</q>.
</p>

<img src="Blog/textGrid4.png" alt="NMF" style="width:400px;">

<h2>Bigrams</h2>
<p>
So far, we looked only and individual words and ignored the arrangement of words
within the texts. For instance: <q>I am wrong.</q> and <q>Am I wrong?</q> have different
meanings, even though they have exactly the same words. So instead of looking
for topics generated by single words, we looked for topics generated by two-word
phrases, called <q>bigrams</q>.
</p>

<p><a href="http://paralign.me/thought"> Click here to explore the topics in greater detail!</a></p>

</body>
</html>
