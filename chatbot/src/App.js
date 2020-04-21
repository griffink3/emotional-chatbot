import React from 'react';
import './App.css';
import { Route, Switch } from 'react-router';
import HomePage from './components/Homepage';

function App() {
  return (
    <div>
      <Switch>
          <Route exact path="/" component={HomePage} />
          {/* <Route component={NotFound} /> */}
      </Switch>
    </div>  
  );
}

export default App;
