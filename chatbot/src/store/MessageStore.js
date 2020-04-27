const addMessage = 'ADD_MESSAGE';
const resetMessages = 'RESET_MESSAGES';
const initialState = { 
    messages: [{ type: 'bot', text: 'Hi, my name is Cooper. Say something!' }] 
};

export const actionCreators = {
  addMessage: (message) => ({ type: addMessage, message }),
  resetMessages: () => ({type: resetMessages})
};

export const reducer = (state, action) => {
  state = state || JSON.parse(JSON.stringify(initialState));

  if (action.type === addMessage) {
    let messages = state.messages;
    messages.push(action.message)
    console.log(action.message)
    return { ...state, message: messages };
  }

  if (action.type === resetMessages) {
    return JSON.parse(JSON.stringify(initialState));
  }

  return state;
};