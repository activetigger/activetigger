import axios from 'axios';
import { useCallback, useState } from 'react';

import config from './config';
import { getAuthHeaders, useAuth } from './useAuth';

/**
 * This hook slices the file into small parts sent as a sequence of
 * short requests — the proxy timeout applies per request, so the total
 * upload time is unbounded. Failed chunks are retried.
 */

// small enough that one chunk goes through well under a 60s proxy timeout
// even on a slow connection (~1 Mbps)
const CHUNK_SIZE = 4 * 1024 * 1024;
const CHUNK_RETRIES = 3;

export interface StagedUpload {
  uploadId: string;
  // sanitized filename under which the server stores the file
  filename: string;
}

export function useChunkedUpload() {
  const { authenticatedUser } = useAuth();
  const [progression, setProgression] = useState<{ loaded?: number; total?: number }>({});
  const [controller, setController] = useState<AbortController | undefined>(undefined);

  const uploadChunked = useCallback(
    async (file: File): Promise<StagedUpload> => {
      const url = config.api.url.replace(/\/$/, '');
      const headers = getAuthHeaders(authenticatedUser)?.headers;
      const ctrl = new AbortController();
      setController(ctrl);
      let uploadId: string | undefined;
      try {
        const totalChunks = Math.max(1, Math.ceil(file.size / CHUNK_SIZE));
        const start = await axios.post(
          `${url}/upload/start`,
          { filename: file.name, total_size: file.size, total_chunks: totalChunks },
          { headers, signal: ctrl.signal },
        );
        uploadId = start.data.upload_id as string;

        for (let index = 0; index < totalChunks; index++) {
          const blob = file.slice(index * CHUNK_SIZE, (index + 1) * CHUNK_SIZE);
          let attempt = 0;
          for (;;) {
            try {
              await axios.postForm(
                `${url}/upload/chunk`,
                { file: new File([blob], file.name) },
                {
                  params: { upload_id: uploadId, index },
                  headers,
                  signal: ctrl.signal,
                  onUploadProgress: ({ loaded }) =>
                    setProgression({
                      loaded: Math.min(index * CHUNK_SIZE + loaded, file.size),
                      total: file.size,
                    }),
                },
              );
              break;
            } catch (error) {
              attempt += 1;
              // do not retry on user cancel or after too many failures
              if (axios.isCancel(error) || attempt >= CHUNK_RETRIES) throw error;
              await new Promise((resolve) => setTimeout(resolve, 1000 * attempt));
            }
          }
        }

        const finish = await axios.post(`${url}/upload/finish`, null, {
          params: { upload_id: uploadId },
          headers,
          signal: ctrl.signal,
        });
        return { uploadId, filename: finish.data.filename as string };
      } catch (error) {
        // drop the server-side session; ignore failures (the server
        // garbage-collects stale sessions anyway)
        if (uploadId) {
          axios.delete(`${url}/upload/${uploadId}`, { headers }).catch(() => undefined);
        }
        throw error;
      } finally {
        setProgression({});
        setController(undefined);
      }
    },
    [authenticatedUser],
  );

  return { uploadChunked, progression, cancel: controller };
}
