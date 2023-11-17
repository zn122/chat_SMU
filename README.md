# Towards the Self-Disclosing Generative Model Capable of Explaining the Rationale on Emotionally Supportive Response 
## new_Emotional-Support-Conversation

by A-Yeon Kim1,+, You-Jin Roh1,+, Sae-Lim Jeong1, Se-Ik Park1, Eun-Seok Oh1 ,
Hye-Jin Hong2, and Jee Hang Lee1,2,* (+These authors contributed equally; *Corresponding author)

Emotionally supportive chatbots built upon the deep generative models have paid much attention due to their efficacy and ubiquity in the treatment of mental health issues. Nonetheless, it seems still unclear that the generated responses are adequate as well as dependable in response to the user’s context. In this context, a recent study proposed a deep generative model that can explain the reason why the chatbot generated such a response with the supplement information e.g., emotion, its intensity and treatment strategy to the user’s input. However, their supplement information was predicted by two separated pre-trained generative models - one for predicting emotion and treatment strategy and the other for predicting the emotion intensity - each of which was trained using the different dataset. This brought about the misalignment between the two pieces of information which has the potential to significantly diminish the capacity to explain the chatbot’s response.  

To this end, we introduce a single neural network-based generative model, called ET2 (emotionally supportive generative model explaining Emotion, inTensity, sTrategy), which can explain the rationale for the generated emotionally supportive responses. Since our proposal was trained with one dataset including the emotion, its intensity and the strategy, the supplement information to explain would be well-aligned as well as consistent between them. To precisely examine it, we evaluated its performance with the test sets. Additionally, we performed the human evaluation on the quality of the model’s explain ability using Mean Opinion Scores (MOSs).  We believe that our proposal could be sufficient to provide users more dependable and consistent emotional support during their interactions through the deep understanding of the user’s context embedded in their conversation.

we propose a **single neural network-based generative model** which can explain the rationale for the generated emotionally supportive responses.

*Details can be found in our paper (with the title above) accepted for publication at ICES 2023. The PDF is available [here](https://drive.google.com/file/d/1W_a_tQrHXOVmadTCzKiv2-ljjtYhCKAD/view?usp=sharing).*

## 한국어 부분 추가
