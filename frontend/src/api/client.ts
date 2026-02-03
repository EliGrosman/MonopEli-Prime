import axios, { type AxiosInstance, type AxiosError } from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * API error response format.
 */
export interface ApiError {
  detail: string;
  code?: string;
}

/**
 * Create configured axios instance.
 */
function createClient(): AxiosInstance {
  const client = axios.create({
    baseURL: `${API_BASE_URL}/api`,
    timeout: 10000,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  // Response interceptor for error handling
  client.interceptors.response.use(
    (response) => response,
    (error: AxiosError<ApiError>) => {
      if (error.response) {
        // Server responded with error
        const message = error.response.data?.detail || 'An error occurred';
        console.error('API Error:', message);
        return Promise.reject(new Error(message));
      } else if (error.request) {
        // Request made but no response
        console.error('Network Error:', error.message);
        return Promise.reject(new Error('Network error - please check your connection'));
      } else {
        // Request setup error
        console.error('Request Error:', error.message);
        return Promise.reject(error);
      }
    }
  );

  return client;
}

export const apiClient = createClient();

/**
 * Add session ID to request headers.
 */
export function setSessionId(sessionId: string | null): void {
  if (sessionId) {
    apiClient.defaults.headers.common['X-Session-ID'] = sessionId;
  } else {
    delete apiClient.defaults.headers.common['X-Session-ID'];
  }
}
