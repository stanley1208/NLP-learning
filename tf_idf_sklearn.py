from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from visual import show_tfidf   # this refers to visual.py in my [repo](https://github.com/MorvanZhou/NLP-Tutorials/)
import numpy as np

docs = [
    "it is a good day, I like to stay here",
    "I am happy to be here",
    "I am bob",
    "it is sunny today",
    "I have a party today",
    "it is a dog and that is a cat",
    "there are dog and cat on the tree",
    "I study hard this morning",
    "today is a good day",
    "tomorrow will be a good day",
    "I like coffee, I like book and I like apple",
    "I do not like it",
    "I am kitty, I like bob",
    "I do not care who like bob, but I like kitty",
    "It is coffee time, bring your cup",
    "At that moment I was the most fearsome weasel in the entire swamp.",
    "The three-year-old girl ran down the beach as the kite flew behind her.",
    "She always had an interesting perspective on why the world must be flat.",
    "As he waited for the shower to warm, he noticed that he could hear water change temperature.",
    "She wanted to be rescued, but only if it was Tuesday and raining.",
    "Lucifer was surprised at the amount of life at Death Valley.",
    "Charles ate the french fries knowing they would be his last meal.",
    "The busker hoped that the people passing by would throw money, but they threw tomatoes instead, so he exchanged his hat for a juicer.",
    "Garlic ice-cream was her favorite.",
    "There aren't enough towels in the world to stop the sewage flowing from his mouth.",
    "He found his art never progressed when he literally used his sweat and tears.",
    "It was at that moment that he learned there are certain parts of the body that you should never Nair.",
    "It was her first experience training a rainbow unicorn.",
    "The swirled lollipop had issues with the pop rock candy.",
    "He ran out of money, so he had to stop playing poker.",
    "As the years pass by, we all know owners look more and more like their dogs.",
    "They throw cabbage that turns your brain into emotional baggage.",
    "She wanted a pet platypus but ended up getting a duck and a ferret instead.",
    "He put heat on the wound to see what would grow.",
    "Going from child, to childish, to childlike is only a matter of time.",
    "He decided water-skiing on a frozen lake wasn’t a good idea.",
    "It's a skateboarding penguin with a sunhat!",
    "Stop waiting for exceptional things to just happen.",
    "The Guinea fowl flies through the air with all the grace of a turtle.",
    "All you need to do is pick up the pen and begin.",
    "The tears of a clown make my lipstick run, but my shower cap is still intact.",
    "They did nothing as the raccoon attacked the lady’s bag of food.",
    "In that instant, everything changed.",
    "He hated that he loved what she hated about hate.",
    "He dreamed of leaving his law firm to open a portable dog wash.",
    "Please tell me you don't work in a morgue.",
    "He liked to play with words in the bathtub.",
    "I made myself a peanut butter sandwich as I didn't want to subsist on veggie crackers.",
    "After exploring the abandoned building, he started to believe in ghosts.",
    "He realized there had been several deaths on this road, but his concern rose when he saw the exact number.",
    "Nancy decided to make the porta-potty her home.",
    "A purple pig and a green donkey flew a kite in the middle of the night and ended up sunburnt.",
    "The sudden rainstorm washed crocodiles into the ocean.",
    "I checked to make sure that he was still alive.",
    "He had decided to accept his fate of accepting his fate.",
    "Three generations with six decades of life experience.",
    "The waitress was not amused when he ordered green eggs and ham.",
    "The golden retriever loved the fireworks each Fourth of July.",
    "He decided to count all the sand on the beach as a hobby.",
    "Beach-combing replaced wine tasting as his new obsession.",
    "The irony of the situation wasn't lost on anyone in the room.",
    "Hang on, my kittens are scratching at the bathtub and they'll upset by the lack of biscuits.",
    "Edith could decide if she should paint her teeth or brush her nails.",
    "Pair your designer cowboy hat with scuba gear for a memorable occasion.",
    "I love eating toasted cheese and tuna sandwiches.",
    "She liked the coffee shop, but the prices were too high for her to justify for herself.",
    "I will get sick from that.",
    "The doctor diagnosed her with diabetes.",
    "I’d like to honor a person.",
    "He was exhausted by the job search, but knew he wouldn't get to relax until he had found a job.",
    "I may be a little bit drunk.",
    "It's so crowded.",
    "Today is a great day.",
    "She only paid six dollars for it, then sold it online for forty-two.",
    "I hate being late more than anything.",
    "I love frozen yogurt.",
    "I do not want to smoke.",
    "African elephants have bigger ears than Asian elephants.",
    "Everyone who got eliminated in red-light green-light got to go inside and eat cookies.",
    "It was a big honor to be asked to give a speech.",
    "I'm going to eat yogurt.",
    "Get away from me, you slimy little worm!",
    "The lake looked brown and foamy, like the water was filled with sand.",
    "Time's up!",
    "Is she choking?",
    "She will switch the light off.",
    "The app wasn't loading for me.",
    "His boss might get angry with his tone.",
    "She was shaking like a chihuahua.",
    "I am extra ticklish for some reason.",
    "I wouldn’t bet on going to the fair today.",
    "His chopsticks were made from bamboo.",
    "What is it?",
    "I won’t phone my friend till Bob arrives.",
    "You have to do it properly.",
    "I feel like blowing something up.",
    "Juan watched TV.",
    "They're almost three times as big as we are.",
    "My brother takes out the trash.",
    "I'll see you tonight at eight.",
    "I don’t like tea.",
    "Can I help you?",

]



vectorizer = TfidfVectorizer()
tf_idf = vectorizer.fit_transform(docs)
print("idf: ", [(n, idf) for idf, n in zip(vectorizer.idf_, vectorizer.get_feature_names())])
print("v2i: ", vectorizer.vocabulary_)

for i in range(9999):
    q = str(input())
    if q=="stop":
        break
    else:
        qtf_idf = vectorizer.transform([q])
        res = cosine_similarity(tf_idf, qtf_idf)
        res = res.ravel().argsort()[-3:]
        print("\ntop 3 docs for '{}':\n{}".format(q, [docs[i] for i in res[::-1]]))


i2v = {i: v for v, i in vectorizer.vocabulary_.items()}
dense_tfidf = tf_idf.todense()
show_tfidf(dense_tfidf, [i2v[i] for i in range(dense_tfidf.shape[1])], "tfidf_sklearn_matrix")