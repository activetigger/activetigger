import { useCallback } from 'react';

import { NotificationData } from '../types';
import { useAppContext } from './useAppContext';

let INCREMENTAL_ID = 1;
const TIME_NOTIFICATION = 10;

export function useNotifications() {
  const { setAppContext } = useAppContext();

  const notify = useCallback(
    (notif: NotificationData) => {
      const id = ++INCREMENTAL_ID;
      const now = new Date();
      setAppContext(
        // in a setter we can use a state modification method
        (state) =>
          // the param is the current state
          ({
            ...state, // here we want to keep the current state object untouched so we spread it
            // but we update the notifications key with it's new value
            notifications: [
              { id, createdAt: now, ...notif },
              ...state.notifications
                .filter((e) => {
                  return (Number(now) - Number(e.createdAt)) / 1000 <= TIME_NOTIFICATION;
                })
                .slice(0, 4),
            ],
          }),
      );
    },
    [setAppContext],
  );

  return { notify };
}
