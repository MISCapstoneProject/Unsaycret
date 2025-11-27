import { Speaker, Session, SpeechLog } from '../types';
import { API_ENDPOINTS } from '../config/api';

class ApiService {
  // Speaker APIs
  async fetchSpeakers(): Promise<Speaker[]> {
    const response = await fetch(API_ENDPOINTS.speakers);
    if (!response.ok) throw new Error('Failed to fetch speakers');
    return response.json();
  }

  async fetchSpeaker(uuid: string): Promise<Speaker> {
    const response = await fetch(API_ENDPOINTS.speaker(uuid));
    if (!response.ok) throw new Error('Failed to fetch speaker');
    return response.json();
  }

  async updateSpeaker(uuid: string, updates: Partial<Speaker>): Promise<void> {
    const response = await fetch(API_ENDPOINTS.speaker(uuid), {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    if (!response.ok) throw new Error('Failed to update speaker');
  }

  async deleteSpeaker(uuid: string): Promise<void> {
    const response = await fetch(API_ENDPOINTS.speaker(uuid), {
      method: 'DELETE',
    });
    if (!response.ok) throw new Error('Failed to delete speaker');
  }

  async verifySpeaker(audioBlob: Blob, threshold: number = 0.4, maxResults: number = 3): Promise<any> {
    const formData = new FormData();
    formData.append('file', audioBlob, 'audio.wav');
    formData.append('threshold', threshold.toString());
    formData.append('max_results', maxResults.toString());

    const response = await fetch(`${API_ENDPOINTS.speakers}/verify`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) throw new Error('Failed to verify speaker');
    return response.json();
  }

  async transferSpeakerVoiceprints(transferData: {
    source_speaker_id: string;
    source_speaker_name: string;
    target_speaker_id: string;
    target_speaker_name: string;
  }): Promise<any> {
    const response = await fetch(`${API_ENDPOINTS.speakers}/transfer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(transferData),
    });
    
    if (!response.ok) throw new Error('Failed to transfer speaker voiceprints');
    return response.json();
  }

  async fetchSpeakerSessions(uuid: string): Promise<Session[]> {
    const response = await fetch(API_ENDPOINTS.speakerSessions(uuid));
    if (!response.ok) throw new Error('Failed to fetch speaker sessions');
    return response.json();
  }

  async fetchSpeakerSpeechLogs(uuid: string): Promise<SpeechLog[]> {
    const response = await fetch(API_ENDPOINTS.speakerSpeechLogs(uuid));
    if (!response.ok) throw new Error('Failed to fetch speaker speech logs');
    return response.json();
  }

  // Session APIs
  async fetchSessions(): Promise<Session[]> {
    const response = await fetch(API_ENDPOINTS.sessions);
    if (!response.ok) throw new Error('Failed to fetch sessions');
    return response.json();
  }

  async createSession(sessionData: {
    title: string;
    session_type?: string;
    participants?: string[];
  }): Promise<Session> {
    const response = await fetch(API_ENDPOINTS.sessions, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(sessionData),
    });
    if (!response.ok) throw new Error('Failed to create session');
    return response.json();
  }

  async updateSession(uuid: string, updates: Partial<Session>): Promise<void> {
    const response = await fetch(API_ENDPOINTS.session(uuid), {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    if (!response.ok) throw new Error('Failed to update session');
  }

  async deleteSession(uuid: string): Promise<void> {
    const response = await fetch(API_ENDPOINTS.session(uuid), {
      method: 'DELETE',
    });
    if (!response.ok) throw new Error('Failed to delete session');
  }

  async fetchSessionSpeechLogs(uuid: string): Promise<SpeechLog[]> {
    const response = await fetch(API_ENDPOINTS.sessionSpeechLogs(uuid));
    if (!response.ok) throw new Error('Failed to fetch session speech logs');
    return response.json();
  }

  // SpeechLog APIs
  async updateSpeechLog(uuid: string, updates: Partial<SpeechLog>): Promise<void> {
    const response = await fetch(API_ENDPOINTS.speechLog(uuid), {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    if (!response.ok) throw new Error('Failed to update speech log');
  }

  async deleteSpeechLog(uuid: string): Promise<void> {
    const response = await fetch(API_ENDPOINTS.speechLog(uuid), {
      method: 'DELETE',
    });
    if (!response.ok) throw new Error('Failed to delete speech log');
  }

  // Audio transcription
  async transcribeAudio(audioBlob: Blob): Promise<any> {
    const formData = new FormData();
    formData.append('file', audioBlob, 'audio.wav');

    const response = await fetch(API_ENDPOINTS.transcribe, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) throw new Error('Failed to transcribe audio');
    return response.json();
  }
}

export const apiService = new ApiService();