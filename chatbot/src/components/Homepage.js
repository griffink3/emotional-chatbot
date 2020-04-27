import React from 'react'
import { Button, Form, Grid, Header, Image, Message, Segment, Modal, Input } from 'semantic-ui-react'
import {bindActionCreators} from "redux";
import { actionCreators as messageActions } from '../store/MessageStore';
import {connect} from "react-redux";
import PageLoader from "./PageLoader";
import axios from 'axios';

class HomePage extends React.Component {
  constructor(props) {
    super(props);
    this._isMounted = false;
  }

  state = {
    isLoading: true,
    currText: '',
    messages: this.props.messages
  }

  componentDidMount() {
    this._isMounted = true;
    this.setState({ isLoading: false });
  }

  componentWillUnmount() {
    this._isMounted = false;
  }

// Helpful Functions

handleChange = (e, { name, value }) => this.setState({ [name]: value });

addUserText = async () => {
    this.props.addMessage({ type: 'user', text: this.state.currText });
    let msg = this.state.currText;
    this.setState({ currText: '' });
    let ret = await axios.get("http://localhost:5000/send?text=" + msg).catch(e => console.error(e));
    if (ret && ret.data) {
        this.props.addMessage({ type: 'bot', text: ret.data });
    } else {
        this.props.addMessage({ type: 'bot', text: 'My brain isn\'t working right now. Sorry!' });
    }
    this.setState({ messages: this.props.messages})
}

// Component Functions

botText = (text) => {
    return (
    <Message style={{ textAlign: 'left', maxWidth: '75%' }}>
        <p>{text}</p>
    </Message>  
    )
}

userText = (text) => {
    return (
    <div style={{ direction: 'rtl' }}>
        <Message style={{ direction: 'ltr', textAlign: 'left', maxWidth: '75%', marginBottom: '10px' }}>
            <p>{text}</p>
        </Message> 
    </div>
    )
}

getMessages = () => {
    return (
    <Segment style={{overflow: 'auto', minHeight: '50%', maxHeight: '50%', marginBottom: '0px' }}>
        {this.props.messages.map(message => {
            if (message.type == 'bot') {
                return this.botText(message.text)
            } else if (message.type == 'user') {
                return this.userText(message.text)
            } 
        })}
        {/* {this.botText('Hi! My name is Cooper.')}
        {this.userText('Hi Cooper. How are you?')}
        <Message style={{ textAlign: 'left', maxWidth: '75%' }}>
            <p>Good. How about you?</p>
        </Message>   
        <div style={{ direction: 'rtl' }}>
        <Message style={{ direction: 'ltr', textAlign: 'left', maxWidth: '75%' }}>
            <p>I'm great. So what do you like to do?</p>
        </Message> 
        </div>  
        <Message style={{ textAlign: 'left', maxWidth: '75%' }}>
            <p>Mostly smoke weed.</p>
        </Message>    
        <div style={{ direction: 'rtl' }}>
        <Message style={{ direction: 'ltr', textAlign: 'left', maxWidth: '75%' }}>
            <p>That's awesome! We should smoke together some time.</p>
        </Message> 
        </div>  
        <Message style={{ textAlign: 'left', maxWidth: '75%' }}>
            <p>I would like that very much.</p>
        </Message>      */}
    </Segment>
    )
}

render() {
    if (this.state.isLoading) {
      return <PageLoader />
    }
    return (
        <Grid textAlign='center' style={{ margin: '0px', height: '110vh', backgroundColor: '#cff4ff' }}>
            <Grid.Column style={{ maxWidth: 450 }}>
                <Header size='huge' style={{ marginTop: '10%', marginBottom: '0%' }}>Cooper</Header>
                <Header size='small' style={{ marginTop: '1%', marginBottom: '10%' }}>The Emotional Chatbot</Header>
                {this.getMessages()}
                <Input 
                    size='large' 
                    name='currText'
                    onChange={this.handleChange}
                    value={this.state.currText}
                    style={{ 
                        minWidth: '100%', 
                        borderWidth: '1px'
                    }} 
                    action={{
                        color: 'teal',
                        content: 'Send',
                        onClick: this.addUserText
                      }}
                    placeholder='Say something...' />
                <Button 
                    size='large' 
                    onClick={this.props.resetMessages} 
                    style={{ marginTop: '10%' }}
                >
                        Reset Messages
                </Button>
            </Grid.Column>
        </Grid>
      )
  }
}

export default connect(
  state => state.message,
  dispatch => bindActionCreators(messageActions, dispatch)
)(HomePage);
