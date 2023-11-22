# Development of a deep learning model for generating explainable and emotionally supportive conversations for mental health self-care

## [2023_ICES]Towards the Self-Disclosing Generative Model Capable of Explaining the Rationale on Emotionally Supportive Response 

by A-Yeon Kim1,+, You-Jin Roh1,+, Sae-Lim Jeong1, Se-Ik Park1, Eun-Seok Oh1 ,
Hye-Jin Hong2, and Jee Hang Lee1,2,* (+These authors contributed equally; *Corresponding author)

Emotionally supportive chatbots built upon the deep generative models have paid much attention due to their efficacy and ubiquity in the treatment of mental health issues. Nonetheless, it seems still unclear that the generated responses are adequate as well as dependable in response to the user’s context. In this context, a recent study proposed a deep generative model that can explain the reason why the chatbot generated such a response with the supplement information e.g., emotion, its intensity and treatment strategy to the user’s input. However, their supplement information was predicted by two separated pre-trained generative models - one for predicting emotion and treatment strategy and the other for predicting the emotion intensity - each of which was trained using the different dataset. This brought about the misalignment between the two pieces of information which has the potential to significantly diminish the capacity to explain the chatbot’s response.  

To this end, we introduce a single neural network-based generative model, called ET2 (emotionally supportive generative model explaining Emotion, inTensity, sTrategy), which can explain the rationale for the generated emotionally supportive responses. Since our proposal was trained with one dataset including the emotion, its intensity and the strategy, the supplement information to explain would be well-aligned as well as consistent between them. To precisely examine it, we evaluated its performance with the test sets. Additionally, we performed the human evaluation on the quality of the model’s explain ability using Mean Opinion Scores (MOSs).  We believe that our proposal could be sufficient to provide users more dependable and consistent emotional support during their interactions through the deep understanding of the user’s context embedded in their conversation.

we propose a **single neural network-based generative model** which can explain the rationale for the generated emotionally supportive responses.

*Details can be found in our paper (with the title above) accepted for publication at ICES 2023. The PDF is available [here](https://drive.google.com/file/d/1W_a_tQrHXOVmadTCzKiv2-ljjtYhCKAD/view?usp=sharing).*

## [2023_KIPS]On the Predictive Model for Emotion Intensity Improving the Efficacy of Emotionally Supportive Chat

Sae-Lim Jeong1,+, You-Jin Roh1,+, Eun-Seok Oh1, A-Yeon Kim1, Hye-Jin Hong2, Jee Hang Lee3,*

A representative emotional dimension model used in the engineering field is Russell's Circumplex model, which expresses emotions by considering two axes: Valence and Arousal. Since each emotion is expressed only as one point, there is a limitation that it is insufficient to express individual differences. In recent studies, the form of an ellipse in which each area of emotion can be expressed was adopted, and this was attempted to be realized. 

An emotional dictionary was used to extract keywords from Korean sentences. Based on the absolute value of the value existing in the dictionary, the importance of words for each morpheme in terms of emotion was ranked. However, in the case of Korean, there are cases where it cannot be called a keyword when contextually grasped because there are many single-letter words when separated into morpheme, so in this case, fast-text was used to replace the keyword. 

When the keyword was extracted from the sentence and then the keyword was projected on the ellipse, we could find the ellipse center distance for each emotion. Using this, it was possible to predict emotions, and after using the difference in probability between top1 emotion and top2 emotion according to the probability value, a 10-point scale could be used to build an integrity area. 

Because emotions were predicted through limited resources and simple distance calculations, there were many shortcomings, and in fact, the results were not good compared to KOBERT, a model that was widely distributed in the past. However, I think this study produced the result of extracting words with high emotional influence within the sentence and was a significant process of suggesting intelligence through the confidence of predicted emotions.

*Details can be found in our paper (with the title above) accepted for publication at KIPS 2023. The PDF is available [here](https://drive.google.com/file/d/1hvabNBjwJsIH3XlQnNggeFFy7M1RxwCC/view?usp=sharing).*

