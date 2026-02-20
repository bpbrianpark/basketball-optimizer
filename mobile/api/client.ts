/**
 * API client for the Basketball Shot Form Analyzer backend.
 * Base URL is read from Settings (AsyncStorage) or constants/config.
 * Stub implementations until MOB-6 (real HTTP).
 */

export interface UploadResponse {
  video_id: string;
}

export interface AnalyzeResponse {
  result_id: string;
}

export interface ResultResponse {
  video_id: string;
  score: number;
  overlay_frames?: string[];
  total_frames?: number;
}

const BACKEND_URL_KEY = 'basketball_optimizer_backend_url';

/**
 * Returns the backend base URL from AsyncStorage (Settings) or default config.
 */
export async function getBaseUrl(): Promise<string> {
  const AsyncStorage = (await import('@react-native-async-storage/async-storage')).default;
  const { DEFAULT_BACKEND_BASE_URL } = await import('@/constants/config');
  const stored = await AsyncStorage.getItem(BACKEND_URL_KEY);
  return (stored && stored.trim()) || DEFAULT_BACKEND_BASE_URL;
}

/**
 * Upload a video file to the backend. Returns video_id.
 * Stub: not implemented until MOB-6.
 */
export async function uploadVideo(_file: { uri: string; name?: string }): Promise<UploadResponse> {
  throw new Error('Not implemented. Real HTTP in MOB-6.');
}

/**
 * Trigger analysis for an uploaded video. Returns result_id.
 * Stub: not implemented until MOB-6.
 */
export async function analyzeVideo(_videoId: string): Promise<AnalyzeResponse> {
  throw new Error('Not implemented. Real HTTP in MOB-6.');
}

/**
 * Fetch analysis result by result_id.
 * Stub: returns mock data until MOB-6.
 */
export async function getResult(_resultId: string): Promise<ResultResponse> {
  // Stub for skeleton: return mock so Results screen can display something.
  return {
    video_id: 'mock',
    score: 0,
    overlay_frames: [],
    total_frames: 0,
  };
}
