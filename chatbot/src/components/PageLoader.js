
import React from 'react';
import { Grid, Segment, Dimmer, Loader } from 'semantic-ui-react'

const PageLoader = () => {
  return (
    <Grid textAlign='center' style={{ height: '125vh', width:'200vh' }} verticalAlign='middle'>
      <Segment style={{ height: '100%', width: '100%' }}>
        <Dimmer active>
          <Loader size='huge'>Loading</Loader>
        </Dimmer>
      </Segment>
    </Grid>
  );
};

export default PageLoader;



