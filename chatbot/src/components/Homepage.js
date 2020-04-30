import React from 'react'
import { Button, Grid, Header, Dropdown, Message, Segment, Input } from 'semantic-ui-react'
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
    mode: 'Positive',
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

handleSelect = (e, { name, value }) => {
    this.setState({ mode: name });
    this.toggleModel(name);
}

addUserText = async () => {
    this.props.addMessage({ type: 'user', text: this.state.currText });
    let msg = this.state.currText;
    this.setState({ currText: '' });
    let ret = await axios.get(`http://localhost:5000/send?text=${msg}`).catch(e => console.error(e));
    if (ret && ret.data) {
        this.props.addMessage({ type: 'bot', text: ret.data });
    } else {
        this.props.addMessage({ type: 'bot', text: 'My brain isn\'t working right now. Sorry!' });
    }
    this.setState({ messages: this.props.messages})
}

toggleModel = async (mode) => {
    await axios.get(`http://localhost:5000/toggle?mode=${mode}`).catch(e => console.error(e));
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
                <Dropdown text={this.state.mode}>
                    <Dropdown.Menu>
                    <Dropdown.Item text='Positive' name='Positive' onClick={this.handleSelect}/>
                    <Dropdown.Item text='Neutral' name='Neutral' onClick={this.handleSelect}/>
                    <Dropdown.Item text='Negative' name='Negative' onClick={this.handleSelect}/>
                    </Dropdown.Menu>
                </Dropdown>
                <Button 
                    size='large' 
                    onClick={this.props.resetMessages} 
                    style={{ marginLeft: '10%', marginTop: '10%' }}
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
