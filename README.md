# Memory for Virtual Companion

## Introduction

*A Step Towards AI-Driven Old Age Care*

This project aims to create an advanced memory module that works seamlessly with large language models. By allowing long-term memory storage, the module enables the AI companion to remember and recall key details about the user's life, creating more personalized and meaningful interactions.

The primary focus is on enhancing old age care, addressing challenges such as social isolation and loneliness among the elderly. The virtual companion acts as a conversational partner, providing companionship and emotional support. Additionally, this project aims to address cognitive decline by recording and regularly updating memories with timestamps. Future versions will include a medication reminder system, providing timely notifications and follow-ups to ensure the treatment plan is being followed.

- [Memory for Virtual Companion](#memory-for-virtual-companion)
  - [Introduction](#introduction)
  - [Memory Flow Process](#memory-flow-process)

## Memory Flow Process

<p align="center">
  <img src="./images/flow.png"/>
  <br>
  <em>Proposed flow for the Memory Module</em>
</p>

**How It Works**:

1. *Relevance Check*:
The system evaluates the user's message to determine if it contains important or relevant information worth storing as a memory.

2. *Memory Extraction*:
If relevant, key details are extracted from the message.

3. *Existence Check*:
The system checks if the memory already exists in its database.

4. *Contradiction Check*: If the memory is new, the system verifies that it does not conflict with existing memories. In case of a conflict, the memory is updated to maintain accuracy and consistency.

This process ensures that the virtual companion can effectively manage and recall user-specific information, providing a more interactive and supportive experience.