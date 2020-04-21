import React from 'react';
import ReactDOM from 'react-dom';
import { Provider } from 'react-redux'
import { ConnectedRouter } from 'connected-react-router'
import './index.css';
import App from './App';
import * as serviceWorker from './serviceWorker';
import { createBrowserHistory } from 'history'
import configureStore from './store/configureStore'
import { persistStore } from 'redux-persist'
import { PersistGate } from 'redux-persist/integration/react'
import PageLoader from './components/PageLoader';

// Create browser history to use in the Redux store
const baseUrl = document.getElementsByTagName('base')[0].getAttribute('href')
const history = createBrowserHistory({ basename: baseUrl })

// Get the application-wide store instance, prepopulating with state from the server where available.
const initialState = window.initialReduxState
const store = configureStore(history, initialState)
const persistor = persistStore(store)

ReactDOM.render((
    <Provider store={store}>
        <PersistGate loading={null} persistor={persistor}>
            <ConnectedRouter history={history}>
                <React.Suspense fallback={<PageLoader />}>
                    <App />
                </React.Suspense>
            </ConnectedRouter>
        </PersistGate>
    </Provider>), document.getElementById('root'));


// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();
