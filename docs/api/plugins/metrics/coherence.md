# Coherence Metrics

::: labchain.plugins.metrics.coherence

## Overview

The Coherence Metrics module in LabChain provides evaluation metrics specifically designed for assessing the quality and interpretability of topic models. These metrics help in understanding how well the generated topics represent the underlying themes in a corpus of documents.

## Available Coherence Metrics

### NPMI (Normalized Pointwise Mutual Information)

The NPMI coherence metric is implemented in the `NPMI` class. It measures the semantic similarity between high-scoring words in the topic based on normalized pointwise mutual information.

#### Usage

```python
from labchain.plugins.metrics.coherence import NPMI
from labchain.base.base_types import XYData

npmi_metric = NPMI(measure='c_npmi', topk=10, processes=1)
score = npmi_metric.evaluate(df, predicted)
```

Parameters:
- `measure`: The coherence measure to use. Default is 'c_npmi'.
- `topk`: The number of top words to consider for each topic. Default is 10.
- `processes`: The number of processes to use for parallel computation. Default is 1.

## How Coherence Metrics Work

Coherence metrics typically work by:

1. Extracting the top N words from each topic.
2. Calculating the semantic similarity between these words based on their co-occurrence in the corpus.
3. Aggregating these similarities to produce a single coherence score for each topic or for the entire model.

The NPMI metric specifically:

1. Calculates the pointwise mutual information (PMI) between pairs of words.
2. Normalizes the PMI scores to account for the frequency of individual words.
3. Averages these normalized scores to produce the final coherence score.

## Comprehensive Example: Evaluating a Topic Model

Here's an example of how to use the NPMI coherence metric to evaluate a topic model:

```python
import pandas as pd
from labchain.plugins.filters.topic_modeling.lda import LDAPlugin
from labchain.plugins.metrics.coherence import NPMI
from labchain.base.base_types import XYData

# Assume we have a DataFrame 'df' with a 'text' column containing our documents
df = pd.DataFrame({'text': ["This is document 1", "This is document 2", "Another document here"]})

# Create XYData object
X_data = XYData(_hash='X_data', _path='/tmp', _value=df['text'].values)

# Create and fit the LDA model
lda_model = LDAPlugin(n_components=5, random_state=42)
lda_model.fit(X_data)

# Get topic-word distributions
topic_word_dist = lda_model.get_topic_word_dist()

# Initialize NPMI metric
npmi_metric = NPMI(measure='c_npmi', topk=10, processes=1)

# Evaluate coherence
coherence_score = npmi_metric.evaluate(df, (None, topic_word_dist, None))

print(f"NPMI Coherence Score: {coherence_score}")
```

This example demonstrates how to:

1. Prepare your text data
2. Create XYData objects for use with LabChain
3. Train an LDA topic model
4. Extract topic-word distributions
5. Initialize and compute the NPMI coherence metric
6. Print the evaluation result

## Best Practices

1. **Multiple Runs**: Topic modeling algorithms often have a random component. Run your model multiple times and average the coherence scores for more stable results.

2. **Number of Topics**: Use coherence metrics to help determine the optimal number of topics for your model. Try different numbers of topics and compare their coherence scores.

3. **Preprocessing**: The quality of your preprocessing can significantly affect coherence scores. Ensure your text is properly cleaned and tokenized.

4. **Interpretation**: Remember that while higher coherence scores generally indicate better topic quality, they should be interpreted in conjunction with qualitative analysis of the topics.

5. **Comparison**: Use coherence metrics to compare different topic modeling approaches (e.g., LDA vs. NMF) on the same dataset.

6. **Domain Knowledge**: Always interpret coherence scores in the context of your domain knowledge and the specific goals of your topic modeling task.

7. **Visualization**: Complement coherence metrics with visualization techniques (like pyLDAvis) to get a better understanding of your topic model results.

8. **Parameter Tuning**: Use coherence scores to guide the tuning of your topic model's parameters (e.g., alpha and beta in LDA).

## Conclusion

The Coherence Metrics module in LabChain provides essential tools for evaluating the quality of topic models. By using these metrics in combination with other LabChain components, you can gain valuable insights into your model's performance and interpretability. The example demonstrates how easy it is to compute and interpret these metrics within the LabChain ecosystem, enabling you to make informed decisions about your topic modeling approach.
