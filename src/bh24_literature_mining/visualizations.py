import matplotlib.pyplot as plt
import math
from wordcloud import WordCloud
from collections import Counter


# Function for creating a histogram of the number of cites/mentions per tool
def plot_histogram(biotools_cites, type="cites"):
    # histogram of number of cites
    num_cites = {len(x["articles"]) for x in biotools_cites}
    log10_values = [math.log10(num) for num in num_cites]
    plt.hist(log10_values, bins=100)
    plt.xlabel("log10(Number of " + type + ")")
    plt.ylabel("Number of tools")
    plt.title("Number of " + type + " per tool")
    plt.show()


# Function to compare mentions and cites
def compare_mentions_cites(biotools_mentions, biotools_cites):
    # Scatter plot of number of mentions vs number of cites
    # create named list of tool mentions and cites
    tool_mentions = {x["name"]: x["articles"] for x in biotools_mentions}
    tool_cites = {x["name"]: x["articles"] for x in biotools_cites}

    # Find the intersection of the two lists
    common_tools = set(tool_mentions.keys()).intersection(set(tool_cites.keys()))

    # create a list of tuples with the number of mentions and cites
    common_mentions = [tool_mentions[tool] for tool in common_tools]
    common_cites = [tool_cites[tool] for tool in common_tools]

    # total number of unique articles
    intersection_articles = []
    total_articles = []
    for tool in common_tools:
        all_articles_mention = [x.pmid for x in tool_mentions[tool]]
        all_articles_cite = [x.pmid for x in tool_cites[tool]]
        all_articles = len(set(all_articles_mention).union(set(all_articles_cite)))
        # intersection of mentions and cites
        common_articles = len(
            set(all_articles_mention).intersection(set(all_articles_cite))
        )
        intersection_articles.append(common_articles)
        total_articles.append(all_articles)

    num_mentions = [len(x) for x in common_mentions]
    num_cites = [len(x) for x in common_cites]

    # create a scatter plot, double log scale
    # point size is proportional to the number of common articles / total articles
    # 50% transparency
    plt.scatter(
        num_mentions,
        num_cites,
        s=[100 * x / y for x, y in zip(intersection_articles, total_articles)],
        alpha=0.5,
    )
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Number of mentions")
    plt.ylabel("Number of citations")
    plt.title("Number of mentions vs number of cites")
    # add x=y line
    plt.plot([1, 1000], [1, 1000], color="red")
    # legend x
    plt.legend(["Size of point proportional to common articles", "x=y"])
    # additional legend with points for different sizes 25%, 50%, 75%, 100%
    plt.scatter([], [], s=25, label="25% common articles", alpha=0.5)
    plt.scatter([], [], s=50, label="50% common articles", alpha=0.5)
    plt.scatter([], [], s=75, label="75% common articles", alpha=0.5)
    plt.scatter([], [], s=100, label="100% common articles", alpha=0.5)
    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title="")
    plt.show()


# Function to make wordcloud of mostly mentioned/cited tools
# Make a word cloud of the most mentioned/cited tools
def wordcloud_most_mentioned(biotools_mentions, max_mentions=1000):
    # Count the number of mentions of each tool
    tool_counter = []
    for tool in biotools_mentions:
        if len(tool["articles"]) > max_mentions:
            tool_counter.append(tool["name"])

    # Create a word cloud
    wordcloud = WordCloud(
        width=800,
        height=800,
        background_color="white",
        stopwords=None,
        min_font_size=10,
    ).generate(" ".join(tool_counter))

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()
